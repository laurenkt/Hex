//
//  LLMClient.swift
//  Hex
//
//  LLM post-processing client using Google Gemini via Application Default Credentials.
//

import Dependencies
import DependenciesMacros
import Foundation
import HexCore

private let logger = HexLog.llm

// MARK: - Client Interface

@DependencyClient
struct LLMClient {
	/// Post-process transcribed text through Gemini. Returns cleaned text.
	var process: @Sendable (_ rawText: String, _ guidance: String, _ timeout: TimeInterval) async throws -> String

	/// Checks whether ADC credentials are available.
	var isAvailable: @Sendable () async -> Bool = { false }
}

extension LLMClient: DependencyKey {
	static var liveValue: Self {
		let live = LLMClientLive()
		return Self(
			process: { rawText, guidance, timeout in
				try await live.process(rawText: rawText, guidance: guidance, timeout: timeout)
			},
			isAvailable: {
				await live.isAvailable()
			}
		)
	}
}

extension DependencyValues {
	var llm: LLMClient {
		get { self[LLMClient.self] }
		set { self[LLMClient.self] = newValue }
	}
}

// MARK: - Live Implementation

private actor LLMClientLive {
	private let session: URLSession = {
		let config = URLSessionConfiguration.default
		config.httpMaximumConnectionsPerHost = 2
		return URLSession(configuration: config)
	}()

	private var cachedToken: CachedToken?

	func isAvailable() -> Bool {
		ADCReader.read() != nil
	}

	func process(rawText: String, guidance: String, timeout: TimeInterval) async throws -> String {
		let token = try await getAccessToken()
		let (statusCode, data) = try await callGemini(rawText: rawText, guidance: guidance, token: token, timeout: timeout)

		// Retry once with a fresh token on 401 (expired token)
		if statusCode == 401 {
			logger.info("Got 401, refreshing token and retrying")
			cachedToken = nil
			let freshToken = try await getAccessToken()
			let (retryStatus, retryData) = try await callGemini(rawText: rawText, guidance: guidance, token: freshToken, timeout: timeout)
			guard retryStatus == 200 else {
				let body = String(data: retryData, encoding: .utf8) ?? ""
				logger.error("Gemini API returned \(retryStatus) after token refresh: \(body, privacy: .private)")
				throw LLMError.apiError(statusCode: retryStatus, body: body)
			}
			return try validateResponse(parseResponse(retryData), inputLength: rawText.count)
		}

		guard statusCode == 200 else {
			let body = String(data: data, encoding: .utf8) ?? ""
			logger.error("Gemini API returned \(statusCode): \(body, privacy: .private)")
			throw LLMError.apiError(statusCode: statusCode, body: body)
		}

		return try validateResponse(parseResponse(data), inputLength: rawText.count)
	}

	private func callGemini(rawText: String, guidance: String, token: String, timeout: TimeInterval) async throws -> (Int, Data) {
		let request = try buildRequest(rawText: rawText, guidance: guidance, token: token, timeout: timeout)
		let (data, response) = try await session.data(for: request)
		guard let http = response as? HTTPURLResponse else {
			throw LLMError.invalidResponse
		}
		return (http.statusCode, data)
	}

	// MARK: - Token Management

	private func getAccessToken() async throws -> String {
		if let cached = cachedToken, cached.isValid {
			return cached.token
		}

		guard let creds = ADCReader.read() else {
			throw LLMError.credentialsNotFound
		}

		let tokenResponse = try await exchangeToken(creds)
		cachedToken = CachedToken(
			token: tokenResponse.accessToken,
			expiry: Date().addingTimeInterval(TimeInterval(tokenResponse.expiresIn - 30))
		)
		logger.info("Refreshed ADC access token")
		return tokenResponse.accessToken
	}

	private func exchangeToken(_ creds: ADCCredentials) async throws -> TokenResponse {
		var request = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
		request.httpMethod = "POST"
		request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")

		let body = [
			"grant_type=refresh_token",
			"refresh_token=\(creds.refreshToken)",
			"client_id=\(creds.clientId)",
			"client_secret=\(creds.clientSecret)",
		].joined(separator: "&")
		request.httpBody = body.data(using: .utf8)

		let (data, _) = try await session.data(for: request)
		return try JSONDecoder().decode(TokenResponse.self, from: data)
	}

	// MARK: - Request Building

	private func buildRequest(rawText: String, guidance: String, token: String, timeout: TimeInterval) throws -> URLRequest {
		let url = URL(string: "https://us-central1-aiplatform.googleapis.com/v1/projects/monzo-ml-exp/locations/us-central1/publishers/google/models/gemini-2.5-flash-lite:generateContent")!
		logger.debug("Using Vertex AI endpoint for project monzo-ml-exp")
		var request = URLRequest(url: url)
		request.httpMethod = "POST"
		request.timeoutInterval = timeout
		request.setValue("application/json", forHTTPHeaderField: "Content-Type")
		request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

		var systemText = "You are a voice transcription cleanup assistant. Fix transcription errors while preserving the speaker's original meaning exactly.\nRules: Fix punctuation, capitalization, and obvious transcription errors. Do NOT add, remove, or rephrase content. Return ONLY the cleaned text."
		if !guidance.isEmpty {
			systemText += "\n\(guidance)"
		}

		let payload: [String: Any] = [
			"systemInstruction": [
				"parts": [["text": systemText]]
			],
			"contents": [
				["role": "user", "parts": [["text": rawText]]]
			],
			"safetySettings": [
				["category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"],
				["category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"],
				["category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"],
				["category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"],
			],
			"generationConfig": [
				"temperature": 0.1,
				"maxOutputTokens": 2048,
			],
		]

		request.httpBody = try JSONSerialization.data(withJSONObject: payload)
		logger.info("System prompt: \(systemText, privacy: .private)")
		logger.info("User text: \(rawText, privacy: .private)")
		return request
	}

	// MARK: - Response Parsing & Validation

	private func parseResponse(_ data: Data) throws -> String {
		let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
		guard
			let candidates = json?["candidates"] as? [[String: Any]],
			let first = candidates.first,
			let content = first["content"] as? [String: Any],
			let parts = content["parts"] as? [[String: Any]],
			let text = parts.first?["text"] as? String
		else {
			throw LLMError.unexpectedResponseFormat
		}
		return text.trimmingCharacters(in: .whitespacesAndNewlines)
	}

	private func validateResponse(_ text: String, inputLength: Int) throws -> String {
		// Reject empty/whitespace-only responses
		guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
			logger.warning("LLM returned empty response, falling back to raw text")
			throw LLMError.emptyResponse
		}

		// Length sanity check: reject if output is wildly different from input
		if inputLength > 0 {
			let ratio = Double(text.count) / Double(inputLength)
			if ratio > 2.0 || ratio < 0.5 {
				logger.warning("LLM output length ratio \(String(format: "%.2f", ratio))x is out of range, falling back to raw text")
				throw LLMError.responseLengthOutOfRange
			}
		}

		return text
	}
}

// MARK: - ADC Reader

private enum ADCReader {
	static func read() -> ADCCredentials? {
		guard let pw = getpwuid(getuid()), let home = pw.pointee.pw_dir else {
			logger.debug("Could not determine real home directory")
			return nil
		}
		let realHome = URL(fileURLWithPath: String(cString: home), isDirectory: true)
		let path = realHome.appendingPathComponent(".config/gcloud/application_default_credentials.json")
		guard let data = try? Data(contentsOf: path) else {
			logger.debug("ADC credentials not found at \(path.path, privacy: .private)")
			return nil
		}
		return try? JSONDecoder().decode(ADCCredentials.self, from: data)
	}
}

// MARK: - Models

private struct ADCCredentials: Decodable {
	let clientId: String
	let clientSecret: String
	let refreshToken: String

	enum CodingKeys: String, CodingKey {
		case clientId = "client_id"
		case clientSecret = "client_secret"
		case refreshToken = "refresh_token"
	}
}

private struct TokenResponse: Decodable {
	let accessToken: String
	let expiresIn: Int

	enum CodingKeys: String, CodingKey {
		case accessToken = "access_token"
		case expiresIn = "expires_in"
	}
}

private struct CachedToken {
	let token: String
	let expiry: Date

	var isValid: Bool { Date() < expiry }
}

enum LLMError: Error, LocalizedError {
	case credentialsNotFound
	case invalidResponse
	case apiError(statusCode: Int, body: String)
	case unexpectedResponseFormat
	case emptyResponse
	case responseLengthOutOfRange

	var errorDescription: String? {
		switch self {
		case .credentialsNotFound:
			"Google Cloud credentials not found. Run 'gcloud auth application-default login' to set up."
		case .invalidResponse:
			"Invalid response from Gemini API."
		case .apiError(let code, _):
			"Gemini API error (HTTP \(code))."
		case .unexpectedResponseFormat:
			"Unexpected response format from Gemini API."
		case .emptyResponse:
			"Gemini returned an empty response."
		case .responseLengthOutOfRange:
			"Gemini response length was too different from input."
		}
	}
}

import Foundation

class Tokenizer: ObservableObject {
    var vocab: [String: Int] = [:]
    var decoder: [Int: String] = [:]

    // Special tokens
    var bosTokenId: Int?
    var eosTokenId: Int?
    var padTokenId: Int?

    init() {}

    func load(tokenizerJSONPath: URL) throws {
        let data = try Data(contentsOf: tokenizerJSONPath)
        // Simplified parsing of tokenizer.json
        // In reality, tokenizer.json is complex.
        // We will assume a simple structure or just basic vocab loading for this demo.
        // Usually, one would use a library like `swift-transformers`.

        // For this implementation, we will try to parse "model" -> "vocab"
        if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
           let model = json["model"] as? [String: Any],
           let vocab = model["vocab"] as? [String: Int] {
            self.vocab = vocab
            self.decoder = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
        } else {
            print("Failed to parse tokenizer.json. Ensure it follows standard Hugging Face format.")
        }

        // Set Qwen default special tokens
        self.bosTokenId = vocab["<|im_start|>"] ?? 151644
        self.eosTokenId = vocab["<|im_end|>"] ?? 151645
        self.padTokenId = vocab["<|endoftext|>"] ?? 151643
    }

    func encode(text: String) -> [Int] {
        // Very basic whitespace tokenization fallback
        // This does NOT implement BPE merging, so it will be incorrect for real inference.
        // TODO: Implement full BPE or use a Swift Tokenizer library.

        var tokens: [Int] = []
        if let bos = bosTokenId { tokens.append(bos) }

        // Naive character/word mapping (only works if words are in vocab)
        // This is just to make the code run without crashing.
        for char in text {
             if let id = vocab[String(char)] {
                 tokens.append(id)
             } else {
                 // Try mapping bytes or fallback
                 // For Qwen, it uses byte fallback.
             }
        }

        if let eos = eosTokenId { tokens.append(eos) }
        return tokens
    }

    func decode(tokens: [Int]) -> String {
        var text = ""
        for token in tokens {
            if let tokenStr = decoder[token] {
                // Handle special tokens and BPE cleanup
                if tokenStr.hasPrefix("<|") { continue } // Skip special tokens for display
                text += tokenStr.replacingOccurrences(of: "Ġ", with: " ") // RoBERTa/GPT style
            }
        }
        return text
    }
}

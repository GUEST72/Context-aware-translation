import torch
import statistics
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, logging as transformers_logging

# Silence harmless BERT initialization warnings
transformers_logging.set_verbosity_error()

class ContextAwareTranslator:
    """
    A dual-model architecture pipeline that utilizes NLLB for context-aware 
    translation generation and mBERT (awesome-align) for dynamic span extraction.
    """
    def __init__(
        self, 
        trans_model_name: str = "facebook/nllb-200-distilled-600M",
        align_model_name: str = "aneuraz/awesome-align-with-co",
        align_layer: int = 8,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Generation Model
        self.trans_model_name = trans_model_name
        self.trans_tokenizer = AutoTokenizer.from_pretrained(self.trans_model_name)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(self.trans_model_name).to(self.device)
        self.trans_tokenizer.src_lang = "eng_Latn"
        self.arb_token_id = self.trans_tokenizer.convert_tokens_to_ids("arb_Arab")
        
        # 2. Initialize Alignment Model
        self.align_model_name = align_model_name
        self.align_tokenizer = AutoTokenizer.from_pretrained(self.align_model_name)
        self.align_model = AutoModel.from_pretrained(self.align_model_name).to(self.device)
        self.align_layer = align_layer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up models and clear cache."""
        if hasattr(self, "trans_model"):
            del self.trans_model
        if hasattr(self, "align_model"):
            del self.align_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_block(self, text):
        """Generates the fused contextual translation for the entire text block."""
        inputs = self.trans_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out_tokens = self.trans_model.generate(**inputs, forced_bos_token_id=self.arb_token_id, max_length=512)
        return self.trans_tokenizer.batch_decode(out_tokens, skip_special_tokens=True)[0]

    def extract_target(self, english_context, arabic_context, english_target):
        """
        Calculates the cross-lingual probability matrix and extracts the target 
        sentence using median-filtered bounding boxes and word-boundary snapping.
        """
        try:
            src_tokens = self.align_tokenizer(english_context, return_tensors="pt", add_special_tokens=False).to(self.device)
            tgt_tokens = self.align_tokenizer(arabic_context, return_tensors="pt", add_special_tokens=False).to(self.device)

            with torch.no_grad():
                out_src = self.align_model(**src_tokens, output_hidden_states=True)
                out_tgt = self.align_model(**tgt_tokens, output_hidden_states=True)
                embed_src = out_src.hidden_states[self.align_layer][0]
                embed_tgt = out_tgt.hidden_states[self.align_layer][0]

            similarity_matrix = torch.matmul(embed_src, embed_tgt.transpose(-1, -2))
            prob_matrix = torch.nn.functional.softmax(similarity_matrix, dim=-1)

            context_ids = src_tokens['input_ids'][0].tolist()
            target_ids = self.align_tokenizer(english_target, return_tensors="pt", add_special_tokens=False)['input_ids'][0].tolist()
            
            target_start_idx = None
            target_len = len(target_ids)
            
            for i in range(len(context_ids) - target_len + 1):
                if context_ids[i : i + target_len] == target_ids:
                    target_start_idx = i
                    break
                    
            if target_start_idx is None:
                return None
                
            aligned_arabic_indices = []
            for i in range(target_start_idx, target_start_idx + target_len):
                aligned_arabic_indices.append(torch.argmax(prob_matrix[i]).item())

            if not aligned_arabic_indices:
                return None

            # Statistical Filter: Mitigate repeated-vocabulary alignment drift
            median_idx = statistics.median(aligned_arabic_indices)
            filtered_indices = [idx for idx in aligned_arabic_indices if abs(idx - median_idx) < 20] # Increased threshold
            
            if not filtered_indices:
                return None
                
            min_idx, max_idx = min(filtered_indices), max(filtered_indices)

            # Boundary Snapping
            while min_idx > 0:
                token_str = self.align_tokenizer.convert_ids_to_tokens(tgt_tokens['input_ids'][0][min_idx].item())
                if token_str.startswith("##"):
                    min_idx -= 1
                else:
                    break
                    
            while max_idx < len(tgt_tokens['input_ids'][0]) - 1:
                next_token_str = self.align_tokenizer.convert_ids_to_tokens(tgt_tokens['input_ids'][0][max_idx + 1].item())
                if next_token_str.startswith("##"):
                    max_idx += 1
                else:
                    break

            span_ids = tgt_tokens['input_ids'][0][min_idx : max_idx + 1]
            return self.align_tokenizer.decode(span_ids, skip_special_tokens=True).strip(" ،.,;")
        except:
            return None

def run_translation():
    translator = ContextAwareTranslator()
    p = "The neural network architecture is quite complex. It requires massive power. Researchers are developing it."
    t = "It requires massive power."
    print(f"\nTarget English: {t}")
    arabic_p = translator.translate_block(p)
    final = translator.extract_target(p, arabic_p, t)
    print(f"Result: {final}")

if __name__ == "__main__":
    run_translation()

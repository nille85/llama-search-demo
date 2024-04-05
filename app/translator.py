from transformers import pipeline


class Translator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def translate(self,text, src_lang, tgt_lang):

        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=1000)
        translation = translator(text)
        return translation[0]["translation_text"]

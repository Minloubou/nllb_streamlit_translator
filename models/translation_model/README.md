# Modèle pour l'exécution locale

Ce dossier est volontairement ignoré par Git, à l'exception de ce fichier README.

Pour une exécution locale, place ici les fichiers générés avec `save_pretrained()`
pour le modèle ET le tokenizer Hugging Face.

Les fichiers peuvent notamment inclure :

- `config.json`
- `generation_config.json`
- `model.safetensors` ou plusieurs fichiers `model-xxxxx-of-xxxxx.safetensors`
- `model.safetensors.index.json` si le modèle est réparti en plusieurs fichiers
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `sentencepiece.bpe.model`
- ainsi que les éventuels fichiers spécifiques générés lors de l'entraînement du tokenizer

La liste exacte dépend de la manière dont le modèle NLLB fine-tuné et son tokenizer
ont été sauvegardés.

Ce projet est destiné à la traduction automatique entre le français et plusieurs
langues africaines du Gabon, notamment le Fang, le Punu et le Myènè, à partir de
modèles NLLB fine-tunés sur des données adaptées à ces langues.

## Utilisation locale

Pour charger un modèle directement depuis ce dossier, configure :

`configs/config.yaml`

avec :

```yaml
model:
  source: "local"
  path_or_repo: "models/translation_model"
  local_files_only: true
# Traducteur des langues gabonaises

Application Streamlit basée sur un modèle **NLLB que j’ai fine-tuné pour la traduction français ↔ fang**.

Le projet porte sur les langues gabonaises à faibles ressources. La version en ligne prend en charge le **français et le fang**. Le punu et le myènè sont prévus pour la suite.

## Démo

[Tester le traducteur](https://nllbapptranslator-t8zmsstoscmtvjbreyngtt.streamlit.app/)

## Modèle et données

Le modèle a été fine-tuné en amont. L’application charge ce modèle et son tokenizer pour traduire les textes, sans relancer l’entraînement.

**Ce dépôt contient le code de l’application, sa configuration et ses tests. Le code d’entraînement n’est pas publié dans ce dépôt. Les données d’entraînement peuvent être mises à disposition sur demande.**

Les poids du modèle sont hébergés séparément dans un dépôt privé sur Hugging Face :

```text
Minloubd/nllb-fang-fr-v5-600M-adafactor
```

Ils ne sont pas inclus dans ce dépôt GitHub. Pour lancer l’application avec ce modèle, un accès au dépôt Hugging Face est nécessaire.

## Fonctionnement

- Interface de traduction avec Streamlit.
- Chargement du modèle avec Hugging Face Transformers.
- Exécution sur CPU, avec une quantification INT8 optionnelle pour réduire la mémoire utilisée.
- Configuration du modèle et des langues dans `configs/config.yaml`.

## Installation et lancement

Depuis la racine du projet, dans un environnement Python 3.11 :

```bash
python -m pip install -r requirements.txt
```

Pour utiliser le modèle privé, ajoutez un token Hugging Face disposant des droits de lecture dans `.streamlit/secrets.toml` :

```toml
HF_TOKEN = "votre_token_hugging_face"
```

Ne publiez pas ce fichier ni votre token sur GitHub. Sur Streamlit Community Cloud, renseignez le token dans les **Secrets** de l’application.

Lancez ensuite l’application :

```bash
python -m streamlit run app.py
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Objectif

Proposer un outil simple de traduction pour des langues gabonaises peu représentées dans les ressources numériques, en associant préparation de données, fine-tuning d’un modèle multilingue et développement d’une application accessible en ligne.

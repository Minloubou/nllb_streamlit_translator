# 🌍 Traducteur automatique des langues gabonaises

Application interactive de traduction automatique développée avec **Streamlit** à partir d'un modèle **NLLB fine-tuné** pour la traduction entre le français et des langues africaines du Gabon.

Le projet s'inscrit dans un travail consacré aux langues à faibles ressources, notamment :

- **Fang**
- **Punu**
- **Myènè**

La version actuellement déployée utilise un modèle fine-tuné pour la traduction **Français ↔ Fang**.

## Fonctionnalités

- Modèle de traduction basé sur **NLLB**
- Chargement avec Hugging Face `AutoTokenizer` et `AutoModelForSeq2SeqLM`
- Prise en charge de modèles stockés localement ou sur **Hugging Face Hub**
- Accès sécurisé aux modèles Hugging Face privés
- Inférence sur CPU
- Quantification dynamique INT8 optionnelle des couches `torch.nn.Linear`
- Gestion des tokens de langue source et cible NLLB
- Configuration centralisée avec un fichier YAML
- Mise en cache du modèle avec Streamlit
- Tests unitaires avec `pytest`
- Application compatible avec **Streamlit Community Cloud**

## Structure du projet

```text
.
├── app.py
├── configs/
│   └── config.yaml
├── models/
│   └── translation_model/
├── src/
│   ├── inference/
│   │   └── translator.py
│   ├── model/
│   │   ├── loader.py
│   │   └── quantization.py
│   └── utils/
│       └── config.py
├── tests/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── requirements-dev.txt
```

## 1. Préparation du modèle

Le modèle et le tokenizer doivent être sauvegardés au format Hugging Face.

Exemple :

```python
model.save_pretrained("models/translation_model")
tokenizer.save_pretrained("models/translation_model")
```

Pour une utilisation entièrement locale, placez ensuite les fichiers dans :

```text
models/translation_model/
```

et configurez `configs/config.yaml` ainsi :

```yaml
model:
  source: "local"
  path_or_repo: "models/translation_model"
  quantize_int8: true
  local_files_only: true
```

Dans le cadre du déploiement de cette application, le modèle est hébergé séparément sur **Hugging Face Hub** afin d'éviter de stocker plusieurs gigaoctets de poids directement dans le dépôt GitHub.

## 2. Configuration du modèle Hugging Face

Le modèle Fang actuellement utilisé est hébergé dans le dépôt :

```text
Minloubd/nllb-fang-fr-v5-600M-adafactor
```

La configuration correspondante dans `configs/config.yaml` est :

```yaml
model:
  source: "huggingface"
  path_or_repo: "Minloubd/nllb-fang-fr-v5-600M-adafactor"
  quantize_int8: true
  local_files_only: false
  trust_remote_code: false
```

Le dépôt Hugging Face contenant le modèle étant privé, l'accès s'effectue avec un **token Hugging Face en lecture seule**.

Ce token ne doit jamais être ajouté directement dans le code ou dans le dépôt GitHub.

Pour une utilisation locale, créez :

```text
.streamlit/secrets.toml
```

avec :

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

Le fichier `secrets.toml` est exclu du suivi Git grâce au `.gitignore`.

Pour Streamlit Community Cloud, le même token doit être ajouté dans les **Secrets** de l'application.

## 3. Configuration des langues

Les langues sont définies dans :

```text
configs/config.yaml
```

Pour le modèle Français ↔ Fang actuellement utilisé :

```yaml
languages:
  Français: "fra_Latn"
  Fang: "fang_Latn"
```

`fra_Latn` correspond au token NLLB du français.

`fang_Latn` correspond au token utilisé pour le Fang lors du fine-tuning.

Les versions destinées au **Punu** et au **Myènè** devront utiliser les tokens exacts définis lors de leur entraînement respectif.

## 4. Installation

Il est recommandé d'utiliser un environnement Python dédié au projet.

Avec Conda :

```bash
conda create -n fang_translator python=3.11 -y
conda activate fang_translator
```

Puis :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Sur certaines configurations macOS utilisant PyTorch 2.2.2, NumPy doit rester en version 1.x. Le projet utilise donc :

```text
numpy==1.26.4
```

## 5. Lancement local

Depuis la racine du projet :

```bash
python -m streamlit run app.py
```

L'application est ensuite accessible, par défaut, à l'adresse :

```text
http://localhost:8501
```

## 6. Exécution des tests

Les tests unitaires peuvent être lancés avec :

```bash
pytest -q
```

Ils vérifient notamment :

- le chargement de la configuration ;
- le fonctionnement du module de quantification ;
- le pipeline d'inférence du traducteur.

## 7. Déploiement sans stocker le modèle sur GitHub

Un modèle NLLB d'environ **600M/660M paramètres** est trop volumineux pour être versionné directement dans un dépôt Git standard.

L'architecture retenue est donc :

```text
GitHub
  |
  | code source
  v
Streamlit Community Cloud
  |
  | authentification avec HF_TOKEN
  v
Hugging Face Hub privé
  |
  v
Modèle NLLB fine-tuné + tokenizer
```

Le dépôt GitHub contient ainsi uniquement :

- le code source ;
- les fichiers de configuration ;
- les tests ;
- la documentation ;
- les dépendances nécessaires au déploiement.

Les poids du modèle restent hébergés sur Hugging Face.

## 8. Quantification INT8

Afin de réduire l'utilisation mémoire lors de l'inférence CPU, le projet permet d'appliquer une quantification dynamique INT8 aux couches linéaires du modèle.

Exemple :

```python
quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

La quantification est principalement destinée à l'inférence sur CPU.

Le modèle original est d'abord chargé en mémoire en virgule flottante avant d'être quantifié. Le pic de consommation mémoire au démarrage peut donc être supérieur à la taille finale du modèle quantifié.

Les performances de traduction doivent toujours être comparées avant et après quantification afin de vérifier qu'aucune dégradation importante de qualité n'est introduite.

## 9. Objectif du projet

L'objectif est de développer des outils de traduction neuronale pour des **langues africaines à faibles ressources**, en particulier des langues gabonaises encore peu représentées dans les grands corpus et modèles multilingues.

Le projet combine :

- constitution et préparation de corpus ;
- fine-tuning de modèles de traduction multilingues ;
- traitement de langues à faibles ressources ;
- évaluation de modèles de Machine Translation ;
- optimisation de l'inférence ;
- développement d'une application interactive ;
- déploiement d'un modèle NLP.

Les travaux portent actuellement sur le **Fang**, avec une extension prévue ou développée vers le **Punu** et le **Myènè**.
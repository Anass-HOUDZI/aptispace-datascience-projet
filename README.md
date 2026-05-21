# Customer Personality Analysis

[![CI Compilation Pipeline](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/actions/workflows/ci.yml/badge.svg)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/actions/workflows/ci.yml)

Bienvenue dans le dépôt de notre projet Data Science !

Afin de maintenir un historique Git propre et éviter les conflits liés aux fichiers générés, les rapports complets (PDF, HTML, Markdown) ainsi que les sources et journaux d'exécution sont désormais compilés automatiquement par notre pipeline d'Intégration Continue (GitHub Actions) et hébergés sous forme de **Releases**.

## 📥 Téléchargements (Dernière Version)

Accédez directement aux derniers artéfacts générés depuis notre dernière release :

- 📄 **[Rapport Complet (PDF)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/rapport.pdf)**
- 🌐 **[Rapport Interactif (HTML)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/rapport.html)**
- 📖 **[Rapport (Markdown GFM)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/README.md)**
- 📚 **[Code Source Complet (ZIP)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/sources.zip)**
- 📓 **[Notebooks Originaux (ZIP)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/notebooks.zip)**
- 🪵 **[Journaux d'Exécution (ZIP)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/logs.zip)**

> **Note :** Si vous venez de faire un *push*, laissez quelques minutes au pipeline GitHub Actions pour recompiler les documents et mettre à jour la release.

## 🛠️ Développement Local

Si vous souhaitez contribuer ou générer les rapports localement (nécessite [Quarto](https://quarto.org/) et [Go-Task](https://taskfile.dev/)) :

```bash
# Compiler et générer tous les formats
task render

# Prévisualisation dynamique
task preview
```

Consultez [le guide d'installation (PDF)](https://github.com/Anass-HOUDZI/aptispace-datascience-projet/releases/latest/download/INSTALL.md) (également disponible dans les Releases) pour configurer votre environnement.

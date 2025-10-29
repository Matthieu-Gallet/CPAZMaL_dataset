#!/usr/bin/env bash
###############################################################################
# Script de configuration de l'environnement virtuel pour dataset_PAZ_TSX
# Usage: ./setup_venv.sh
###############################################################################

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="venv_dataset_paz"
PYTHON_VERSION="python3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Environment - dataset_PAZ_TSX${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Vérifier que Python est installé
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Erreur: $PYTHON_VERSION n'est pas installé${NC}"
    echo "Veuillez installer Python 3.8 ou supérieur"
    exit 1
fi

PYTHON_CMD=$(command -v $PYTHON_VERSION)
echo -e "${GREEN}✓${NC} Python trouvé: $PYTHON_CMD"
$PYTHON_CMD --version

# Vérifier si l'environnement virtuel existe déjà
if [ -d "$PROJECT_DIR/$VENV_NAME" ]; then
    echo ""
    echo -e "${YELLOW}⚠ L'environnement virtuel '$VENV_NAME' existe déjà${NC}"
    read -p "Voulez-vous le recréer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Suppression de l'ancien environnement...${NC}"
        rm -rf "$PROJECT_DIR/$VENV_NAME"
    else
        echo -e "${GREEN}Mise à jour de l'environnement existant...${NC}"
        source "$PROJECT_DIR/$VENV_NAME/bin/activate"
        pip install --upgrade pip
        pip install -r "$PROJECT_DIR/requirements.txt"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ Environnement mis à jour avec succès  ${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Pour activer l'environnement:"
        echo "  Bash/Zsh: source $VENV_NAME/bin/activate"
        echo "  Fish:     source $VENV_NAME/bin/activate.fish"
        echo ""
        echo "Pour désactiver:"
        echo "  deactivate"
        exit 0
    fi
fi

# Créer l'environnement virtuel
echo ""
echo -e "${GREEN}Création de l'environnement virtuel '$VENV_NAME'...${NC}"
$PYTHON_CMD -m venv "$PROJECT_DIR/$VENV_NAME"

# Activer l'environnement
echo -e "${GREEN}Activation de l'environnement virtuel...${NC}"
source "$PROJECT_DIR/$VENV_NAME/bin/activate"

# Mettre à jour pip
echo ""
echo -e "${GREEN}Mise à jour de pip...${NC}"
pip install --upgrade pip

# Installer les dépendances
echo ""
echo -e "${GREEN}Installation des dépendances depuis requirements.txt...${NC}"
echo ""

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo -e "${RED}Erreur: requirements.txt non trouvé dans $PROJECT_DIR${NC}"
    exit 1
fi

# Vérification de l'installation
echo ""
echo -e "${GREEN}Vérification des packages installés...${NC}"
echo ""

# Liste des packages critiques à vérifier
CRITICAL_PACKAGES=("numpy" "h5py" "rasterio" "tqdm" "joblib" "pandas" "scipy")

ALL_OK=true
for package in "${CRITICAL_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep "Version:" | cut -d " " -f 2)
        echo -e "${GREEN}✓${NC} $package ($VERSION)"
    else
        echo -e "${RED}✗${NC} $package - NON INSTALLÉ"
        ALL_OK=false
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✓ Installation réussie!${NC}"
else
    echo -e "${RED}⚠ Certains packages sont manquants${NC}"
    echo "Essayez de les installer manuellement avec:"
    echo "  pip install <package_name>"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Instructions pour l'utilisateur
echo "Instructions d'utilisation:"
echo ""
echo "1. Activer l'environnement:"
if [ -n "$FISH_VERSION" ]; then
    echo "   source $VENV_NAME/bin/activate.fish"
else
    echo "   source $VENV_NAME/bin/activate"
fi
echo ""
echo "2. Exécuter les scripts:"
echo "   python script/prepare_hdf5/create_HDF5_dataset.py"
echo "   python script/load_dataset.py"
echo ""
echo "3. Désactiver l'environnement:"
echo "   deactivate"
echo ""
echo -e "${GREEN}Environnement fonctionnel${NC}"

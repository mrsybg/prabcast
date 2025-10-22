#!/usr/bin/env python3
"""
Migration Script für Session State Refactoring.
Ersetzt alte Session State Zugriffe durch neue strukturierte API.
"""

import os
import re
from pathlib import Path

# Definiere die Ersetzungsregeln
REPLACEMENTS = [
    # Session State Zugriffe
    (r'st\.session_state\.df\b', 'state.data.df'),
    (r'st\.session_state\.date_column\b', 'state.data.date_column'),
    (r'st\.session_state\.selected_products_in_data\b', 'state.data.selected_products'),
    (r'st\.session_state\.start_date\b', 'state.data.start_date'),
    (r'st\.session_state\.end_date\b', 'state.data.end_date'),
    (r'st\.session_state\.start_date_selected\b', 'state.data.start_date_selected'),
    (r'st\.session_state\.end_date_selected\b', 'state.data.end_date_selected'),
    (r'st\.session_state\.ready_for_processing\b', 'state.ready_for_processing'),
    
    # Forecast State
    (r'st\.session_state\.multivariate_data\b', 'state.forecast.multivariate_data'),
    (r'st\.session_state\.saved_models_for_complex\b', 'state.forecast.saved_models'),
    (r'st\.session_state\.forecast_complex_results\b', 'state.forecast.forecast_complex_results'),
    (r'st\.session_state\.forecast_complex_target\b', 'state.forecast.forecast_complex_target'),
    (r'st\.session_state\.forecast_status\b', 'state.forecast.forecast_status'),
    (r'st\.session_state\.saved_model_status\b', 'state.forecast.saved_model_status'),
    
    # Get-Zugriffe
    (r"st\.session_state\.get\(['\"]ready_for_processing['\"]\s*,\s*False\)", 'state.ready_for_processing'),
    (r"st\.session_state\.get\(['\"]df['\"]\)", 'state.data.df'),
]

def add_import_if_needed(content):
    """Fügt Import hinzu, falls nicht vorhanden."""
    if 'from setup_module.session_state import get_app_state' not in content:
        # Füge nach dem ersten from setup_module.helpers import hinzu
        import_pattern = r'(from setup_module\.helpers import \*)'
        replacement = r'\1\nfrom setup_module.session_state import get_app_state'
        content = re.sub(import_pattern, replacement, content, count=1)
    return content

def add_state_initialization(content):
    """Fügt state = get_app_state() am Anfang der display_tab Funktion hinzu."""
    # Suche nach def display_tab():
    pattern = r'(def display_tab\(\):)\s*\n'
    
    # Prüfe ob bereits state = get_app_state() vorhanden
    if 'state = get_app_state()' not in content:
        replacement = r'\1\n    """Tab mit strukturiertem Session State Management."""\n    state = get_app_state()\n    \n'
        content = re.sub(pattern, replacement, content, count=1)
    
    return content

def add_ready_check(content):
    """Fügt Ready-Check hinzu falls nicht vorhanden."""
    # Suche nach def display_tab(): und prüfe ob danach bereits ein Check vorhanden
    if 'state = get_app_state()' in content and 'if not state.ready_for_processing:' not in content:
        # Füge Check nach state Initialisierung hinzu (aber nur in relevanten Tabs)
        pattern = r'(state = get_app_state\(\)\s*\n)'
        replacement = r'\1    \n    if not state.ready_for_processing:\n        st.warning("Bitte zuerst Daten hochladen und konfigurieren.")\n        return\n'
        content = re.sub(pattern, replacement, content, count=1)
    
    return content

def migrate_file(filepath):
    """Migriert eine einzelne Datei."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Füge Imports hinzu
    content = add_import_if_needed(content)
    
    # Füge state Initialisierung hinzu
    content = add_state_initialization(content)
    
    # Wende alle Ersetzungen an
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    
    # Spezialfall für upload.py und layout.py - kein ready_check
    if not str(filepath).endswith('upload.py') and not str(filepath).endswith('layout.py') and not str(filepath).endswith('glossar.py'):
        pass  # Bereits im add_state_initialization behandelt
    
    # Schreibe nur zurück wenn Änderungen
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Migrated")
        return True
    else:
        print(f"  ⏭️  No changes needed")
        return False

def main():
    """Hauptfunktion für Migration."""
    tabs_dir = Path(__file__).parent.parent / 'tabs'
    
    print("=" * 60)
    print("Session State Migration")
    print("=" * 60)
    
    # Migriere alle Tab-Dateien
    tab_files = list(tabs_dir.glob('*.py'))
    tab_files = [f for f in tab_files if f.name != '__init__.py']
    
    migrated_count = 0
    for tab_file in sorted(tab_files):
        if migrate_file(tab_file):
            migrated_count += 1
    
    print("=" * 60)
    print(f"Migration complete: {migrated_count}/{len(tab_files)} files updated")
    print("=" * 60)

if __name__ == '__main__':
    main()

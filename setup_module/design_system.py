"""
Design System für PrABCast
===========================
Zentrale UI-Komponenten für konsistente Benutzeroberfläche

Dieses Modul stellt standardisierte UI-Komponenten bereit, um eine
einheitliche und professionelle Benutzeroberfläche zu gewährleisten.
"""

import streamlit as st
from typing import Optional, Callable, Any


class UIComponents:
    """
    Zentrale Klasse für UI-Komponenten mit konsistenten Styles
    
    Enthält:
    - Icons für verschiedene Kontexte
    - Farbschema (Streamlit-kompatibel)
    - Standardisierte Buttons
    - Section-Headers mit optionaler Hilfe
    - Status-Indikatoren
    """
    
    # ============================================================================
    # ICONS - Konsistente Emojis für die gesamte Anwendung
    # ============================================================================
    
    ICONS = {
        # Navigation & Workflow
        'upload': '📁',
        'analysis': '📊',
        'forecast': '🎯',
        'model': '🔬',
        'data': '📈',
        'settings': '⚙️',
        'dashboard': '📉',
        
        # Status & Feedback
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'help': '❓',
        'tip': '💡',
        
        # Actions
        'play': '▶️',
        'stop': '⏹️',
        'refresh': '🔄',
        'download': '📥',
        'save': '💾',
        'delete': '🗑️',
        'back': '↩️',
        
        # Quality & Validation
        'check': '✔️',
        'cross': '✖️',
        'star': '⭐',
        'time': '⏱️',
        'calendar': '📅',
        'chart': '📊',
        
        # Special
        'rocket': '🚀',
        'fire': '🔥',
        'target': '🎯',
        'magic': '✨',
        'lock': '🔒',
        'unlock': '🔓'
    }
    
    # ============================================================================
    # COLORS - Streamlit-kompatibles Farbschema
    # ============================================================================
    
    COLORS = {
        'primary': '#FF4B4B',      # Streamlit Rot
        'success': '#21C354',      # Grün
        'warning': '#FFA500',      # Orange
        'error': '#FF4B4B',        # Rot
        'info': '#4A9EFF',         # Blau
        'neutral': '#808495'       # Grau
    }
    
    # ============================================================================
    # BUTTONS - Standardisierte Button-Komponenten
    # ============================================================================
    
    @staticmethod
    def primary_button(
        label: str,
        key: Optional[str] = None,
        icon: Optional[str] = None,
        disabled: bool = False,
        use_container_width: bool = True,
        help: Optional[str] = None,
        use_icons: bool = False
    ) -> bool:
        """
        Haupt-Action-Button mit konsistentem Styling
        
        Args:
            label: Button-Text
            key: Eindeutiger Key für Streamlit
            icon: Optional - Icon-Schlüssel aus ICONS dict
            disabled: Button deaktiviert
            use_container_width: Button auf volle Breite
            help: Tooltip-Text
            use_icons: Icons anzeigen (default: False für professionelles Design)
            
        Returns:
            True wenn Button geklickt wurde
            
        Example:
            if UI.primary_button("Analyse starten", key="analysis_btn"):
                run_analysis()
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}"
        else:
            display_label = label
        
        return st.button(
            display_label,
            key=key,
            type="primary",
            disabled=disabled,
            use_container_width=use_container_width,
            help=help
        )
    
    @staticmethod
    def secondary_button(
        label: str,
        key: Optional[str] = None,
        icon: Optional[str] = None,
        disabled: bool = False,
        use_container_width: bool = False,
        help: Optional[str] = None,
        use_icons: bool = False
    ) -> bool:
        """
        Sekundärer Button für weniger wichtige Aktionen
        
        Args:
            label: Button-Text
            key: Eindeutiger Key
            icon: Optional - Icon-Schlüssel
            disabled: Button deaktiviert
            use_container_width: Button auf volle Breite
            help: Tooltip-Text
            use_icons: Icons anzeigen (default: False)
            
        Returns:
            True wenn Button geklickt wurde
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}".strip()
        else:
            display_label = label
        
        return st.button(
            display_label,
            key=key,
            type="secondary",
            disabled=disabled,
            use_container_width=use_container_width,
            help=help
        )
    
    # ============================================================================
    # HEADERS & SECTIONS
    # ============================================================================
    
    @staticmethod
    def section_header(
        title: str,
        icon: Optional[str] = None,
        help_text: Optional[str] = None,
        divider: bool = True,
        use_icons: bool = False
    ):
        """
        Standardisierter Section-Header mit optionaler Hilfe
        
        Args:
            title: Header-Text
            icon: Optional - Icon-Schlüssel aus ICONS dict (nicht verwendet wenn use_icons=False)
            help_text: Optional - Hilfetext wird als Caption unter Header angezeigt
            divider: Trennlinie unter Header anzeigen
            use_icons: Icons anzeigen (default: False)
            
        Example:
            UI.section_header("Modellauswahl", help_text="Wählen Sie ein Prognosemodell")
        """
        # Header-Text zusammenstellen
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            header_text = f"{icon_str} {title}".strip()
        else:
            header_text = title
        
        st.subheader(header_text)
        
        # Hilfetext als Caption direkt unter Header
        if help_text:
            st.caption(help_text)
        
        if divider:
            st.divider()
    
    @staticmethod
    def page_header(
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Haupt-Page-Header für Tab-Überschriften
        
        Args:
            title: Haupt-Titel
            subtitle: Optional - Untertitel/Beschreibung
            icon: Optional - Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            header_text = f"{icon_str} {title}".strip()
        else:
            header_text = title
        
        st.header(header_text)
        
        if subtitle:
            st.caption(subtitle)
        
        st.divider()
    
    # ============================================================================
    # STATUS MESSAGES
    # ============================================================================
    
    @staticmethod
    def success_message(
        message: str,
        icon: str = 'success',
        expandable_details: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Erfolgs-Nachricht mit optionalen Details
        
        Args:
            message: Haupt-Nachricht
            icon: Icon-Schlüssel (default: success, nicht verwendet wenn use_icons=False)
            expandable_details: Optional - Zusätzliche Details in Expander
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons:
            icon_str = UIComponents.ICONS.get(icon, UIComponents.ICONS['success'])
            st.success(f"{icon_str} {message}")
        else:
            st.success(message)
        
        if expandable_details:
            with st.expander("Details anzeigen"):
                st.write(expandable_details)
    
    @staticmethod
    def error_message(
        message: str,
        details: Optional[str] = None,
        show_recovery: bool = False,
        on_retry: Optional[Callable] = None,
        on_back: Optional[Callable] = None,
        use_icons: bool = False
    ):
        """
        Fehler-Nachricht mit optionalen Recovery-Optionen
        
        Args:
            message: Haupt-Fehlermeldung
            details: Optional - Technische Details
            show_recovery: Recovery-Buttons anzeigen
            on_retry: Callback für "Neu versuchen" Button
            on_back: Callback für "Zurück" Button
            use_icons: Icons anzeigen (default: False)
        """
        st.error(message)
        
        if details:
            with st.expander("Technische Details"):
                st.code(details)
        
        if show_recovery:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Zurück", key="error_back"):
                    if on_back:
                        on_back()
            with col2:
                if st.button("Neu versuchen", key="error_retry"):
                    if on_retry:
                        on_retry()
    
    @staticmethod
    def warning_message(
        message: str,
        icon: str = 'warning',
        details: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Warnung mit optionalen Details
        
        Args:
            message: Warnungs-Text
            icon: Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            details: Optional - Zusätzliche Informationen
            use_icons: Icons anzeigen (default: False)
        """
        st.warning(message)
        
        if details:
            with st.expander("Mehr Informationen"):
                st.write(details)
    
    @staticmethod
    def info_message(
        message: str,
        icon: str = 'info',
        collapsible: bool = False,
        use_icons: bool = False
    ):
        """
        Info-Nachricht
        
        Args:
            message: Info-Text
            icon: Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            collapsible: Als Expander anzeigen
            use_icons: Icons anzeigen (default: False)
        """
        if collapsible:
            with st.expander("Information"):
                st.write(message)
        else:
            st.info(message)
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def get_icon(key: str, fallback: str = '') -> str:
        """
        Hole Icon nach Schlüssel
        
        Args:
            key: Icon-Schlüssel
            fallback: Fallback wenn Key nicht existiert
            
        Returns:
            Icon-String (Emoji)
        """
        return UIComponents.ICONS.get(key, fallback)
    
    @staticmethod
    def get_color(key: str) -> str:
        """
        Hole Farbe nach Schlüssel
        
        Args:
            key: Farb-Schlüssel
            
        Returns:
            Hex-Farbcode
        """
        return UIComponents.COLORS.get(key, UIComponents.COLORS['neutral'])
    
    @staticmethod
    def status_badge(
        status: str,
        label: Optional[str] = None
    ) -> str:
        """
        Generiere Status-Badge
        
        Args:
            status: 'success', 'error', 'warning', 'info'
            label: Optional - Text neben Icon
            
        Returns:
            Formatierter Status-String
        """
        icon_map = {
            'success': UIComponents.ICONS['success'],
            'error': UIComponents.ICONS['error'],
            'warning': UIComponents.ICONS['warning'],
            'info': UIComponents.ICONS['info'],
            'check': UIComponents.ICONS['check'],
            'cross': UIComponents.ICONS['cross']
        }
        
        icon = icon_map.get(status, UIComponents.ICONS['info'])
        
        if label:
            return f"{icon} {label}"
        return icon
    
    @staticmethod
    def metric_card(
        label: str,
        value: Any,
        delta: Optional[Any] = None,
        help_text: Optional[str] = None,
        icon: Optional[str] = None,
        use_icons: bool = False
    ):
        """
        Anzeige einer Metrik mit optionalem Delta
        
        Args:
            label: Metrik-Bezeichnung
            value: Aktueller Wert
            delta: Optional - Änderung/Vergleich
            help_text: Optional - Tooltip
            icon: Optional - Icon-Schlüssel (nicht verwendet wenn use_icons=False)
            use_icons: Icons anzeigen (default: False)
        """
        if use_icons and icon:
            icon_str = UIComponents.ICONS.get(icon, '')
            display_label = f"{icon_str} {label}".strip()
        else:
            display_label = label
        
        st.metric(
            label=display_label,
            value=value,
            delta=delta,
            help=help_text
        )


# Shortcut für einfacheren Zugriff
UI = UIComponents


# ============================================================================
# MODULE-LEVEL FUNCTIONS (convenience)
# ============================================================================

def get_icon(key: str, fallback: str = '') -> str:
    """Convenience function for getting icons"""
    return UI.get_icon(key, fallback)


def get_color(key: str) -> str:
    """Convenience function for getting colors"""
    return UI.get_color(key)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import plotly.express as px
import os
import re
import json
import copy
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RheologicalAnalyzer:
    """
    Analyzes rheological data by treating every measurement series individually.
    Supports locale settings for number formats (decimal/thousands separators).
    """
    def __init__(self):
        """Initializes the analyzer with empty data structures."""
        self.datasets = {}
        self.original_datasets = {}
        self.fit_results = {}
        self.fit_ranges = {}
        
    def cross_model(self, shear_rate, eta_0, eta_inf, lambda_c, n):
        """Cross Model for viscosity fitting."""
        return eta_inf + (eta_0 - eta_inf) / (1 + (lambda_c * shear_rate)**n)
    
    # --- DATA LOADING AND PARSING ---
    def load_data(self, file_paths, locale_settings=None):
        """
        Loads all measurement series as individual datasets.
        
        Args:
            file_paths (list or str): Path(s) to the CSV files.
            locale_settings (dict, optional): Defines number format. 
                Defaults to German format: {'decimal': ',', 'thousands': '.'}.
                Example for US/UK: {'decimal': '.', 'thousands': ','}.
        """
        if isinstance(file_paths, str): file_paths = [file_paths]
        
        # Set default locale to German if not provided
        if locale_settings is None:
            locale_settings = {'decimal': ',', 'thousands': '.'}
        
        all_datasets = []
        for filepath in file_paths:
            # Pass the settings down to the parser
            all_datasets.extend(self.parse_csv_file(filepath, locale_settings))
        
        self.datasets = {ds['dataset_id']: ds for ds in all_datasets}
        self.original_datasets = copy.deepcopy(self.datasets)
        
        print(f"Loaded {len(self.datasets)} individual measurement series.")

    def parse_csv_file(self, filepath, locale_settings):
        """Parses a CSV file using specified locale settings."""
        datasets = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            sections = content.split('Datenreihen-Informationen')
            for i, section in enumerate(sections[1:], 1):
                dataset = self._parse_single_dataset(section, locale_settings)
                if dataset is not None:
                    dataset['file_source'] = os.path.basename(filepath)
                    dataset['dataset_id'] = f"{os.path.basename(filepath)}_{i}"
                    datasets.append(dataset)
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
        return datasets
    
    def _parse_single_dataset(self, section_text, locale_settings):
        """Parses a single dataset section using specified locale settings."""
        lines = section_text.strip().split('\n')
        metadata, data_start_idx = {}, None
        for i, line in enumerate(lines):
            if 'Name:' in line: metadata['name'] = line.split('","')[3].strip('"')
            elif 'Probe:' in line: metadata['sample'] = line.split('","')[3].strip('"')
            elif 'Datum/Zeit:' in line: metadata['datetime'] = line.split('","')[3].strip('"')
            elif 'Messpkt.' in line and 'Scherrate' in line: data_start_idx = i; break
        if data_start_idx is None: return None
        
        metadata['concentration'] = self._extract_concentration(metadata.get('sample', ''))
        
        data_lines = [line for line in lines[data_start_idx + 2:] if line.strip() and not line.startswith('Datenreihen') and not (line.startswith('"') and line.count('","') < 3)]
        if not data_lines: return None

        # ✅ NEUE LOGIK: Ländereinstellungen für Zahlen anwenden
        decimal_sep = locale_settings.get('decimal', ',')
        thousands_sep = locale_settings.get('thousands', '.')
        
        def clean_number(s):
            """Converts a string to a float using the specified locale."""
            s_no_thousands = s.replace(thousands_sep, '')
            s_decimal_dot = s_no_thousands.replace(decimal_sep, '.')
            return float(s_decimal_dot)

        data_rows = []
        for line in data_lines:
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) >= 6:
                try:
                    row = {
                        'point': int(parts[0]),
                        'shear_rate': clean_number(parts[1]),
                        'shear_stress': clean_number(parts[2]),
                        'viscosity': clean_number(parts[3]),
                        'rotation_speed': clean_number(parts[4]),
                        'torque': clean_number(parts[5]),
                        'status': parts[6] if len(parts) > 6 else ''
                    }
                    data_rows.append(row)
                except (ValueError, IndexError):
                    continue
        
        if not data_rows: return None
        return {'metadata': metadata, 'data': pd.DataFrame(data_rows)}

    def _extract_concentration(self, sample_name):
        """Extracts concentration value from a sample name string using regex."""
        if 'Reines' in sample_name or not sample_name.strip(): return 0.0
        for pattern in [r'(\d+\.?\d*)\s*g/L', r'(\d+\.?\d*)\s*g\s*/\s*L', r'(\d+\.?\d*)\s*gL']:
            match = re.search(pattern, sample_name, re.IGNORECASE)
            if match: return float(match.group(1))
        return None


    
    # Ersetzen Sie diese Methode
    def interactively_select_data_range(self, filename="datarange_filter.json"):
        """Interactively sets a shear rate range and stores it for reporting."""
        self.fit_ranges = {} # Reset ranges for each run
        filtered_datasets = {}
        
        if os.path.exists(filename) and input(f"Filter file '{filename}' found. Load? (y/n): ").lower() == 'y':
            with open(filename, 'r') as f: 
                self.fit_ranges = json.load(f) # Lade und speichere die Grenzen
            print(f"Filter loaded from '{filename}'.")
            for ds_id, dataset in self.datasets.items():
                df = dataset['data']
                if ds_id in self.fit_ranges:
                    x_min, x_max = self.fit_ranges[ds_id]
                    filtered_df = df[(df['shear_rate'] >= x_min) & (df['shear_rate'] <= x_max)]
                    filtered_datasets[ds_id] = {'metadata': dataset['metadata'], 'data': filtered_df}
                else:
                    filtered_datasets[ds_id] = dataset
            return filtered_datasets
    
        print("\n--- Interactive selection of data range for fitting ---")
        chosen_ranges = {}
        for ds_id, dataset in sorted(self.datasets.items()):
            df, conc = dataset['data'], dataset['metadata'].get('concentration', 'N/A')
            fig = px.scatter(df, x='shear_rate', y='viscosity', log_x=True, log_y=True, title=f"<b>ID: {ds_id}</b> (Conc: {conc} g/L)")
            fig.show()
            while True:
                try:
                    x_min_str = input(f"  Min shear rate for {ds_id} (Enter for min): ")
                    x_max_str = input(f"  Max shear rate for {ds_id} (Enter for max): ")
                    x_min = float(x_min_str) if x_min_str.strip() else -np.inf
                    x_max = float(x_max_str) if x_max_str.strip() else np.inf
                    chosen_ranges[ds_id] = [x_min, x_max]
                    filtered_df = df[(df['shear_rate'] >= x_min) & (df['shear_rate'] <= x_max)]
                    print(f"--> Selected {len(filtered_df)} of {len(df)} points.\n")
                    filtered_datasets[ds_id] = {'metadata': dataset['metadata'], 'data': filtered_df}
                    break
                except ValueError: print("Invalid input. Please enter a number.")
        
        self.fit_ranges = chosen_ranges # Speichere die interaktiv gewählten Grenzen
        if self.fit_ranges and input(f"Save filter settings to '{filename}'? (y/n): ").lower() == 'y':
            with open(filename, 'w') as f: json.dump(self.fit_ranges, f, indent=4)
            print("Filter saved.")
        return filtered_datasets

    def estimate_initial_parameters(self, shear_rate, viscosity):
        """Estimates initial parameters for the Cross model fit."""
        if len(viscosity) < 2: return [0, 0, 1, 1]
        return [max(viscosity), min(viscosity), 1.0 / np.mean(shear_rate) if np.mean(shear_rate) > 0 else 1.0, 0.8]

    def fit_cross_model(self):
        """Fits the Cross model to each individual dataset."""
        self.fit_results = {}
        fig, axes = plt.subplots(2, 2, figsize=(15, 12)); fig.suptitle('Cross Model Fitting Results', fontsize=16)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.datasets)))
        default_bounds = ([0, 0, 1e-6, 0.1], [np.inf, np.inf, np.inf, 2.0])
        
        all_conc, all_eta0, all_eta0_err, all_r2, all_n, all_n_err = [], [], [], [], [], []

        for i, (ds_id, dataset) in enumerate(sorted(self.datasets.items())):
            df = dataset['data']
            if len(df) < 5:
                print(f"Skipping fit for {ds_id}: insufficient data points ({len(df)}).")
                continue
            shear_rate, viscosity = df['shear_rate'].values, df['viscosity'].values
            p0 = self.estimate_initial_parameters(shear_rate, viscosity)
            try:
                popt, pcov = curve_fit(self.cross_model, shear_rate, viscosity, p0=p0, bounds=default_bounds, maxfev=5000)
                y_pred = self.cross_model(shear_rate, *popt)
                r_squared = 1 - (np.sum((viscosity - y_pred)**2) / np.sum((viscosity - np.mean(viscosity))**2))
                param_errors = np.sqrt(np.diag(pcov))
                self.fit_results[ds_id] = {'eta_0': popt[0], 'eta_inf': popt[1], 'lambda_c': popt[2], 'n': popt[3], 'eta_0_error': param_errors[0], 'eta_inf_error': param_errors[1], 'lambda_c_error': param_errors[2], 'n_error': param_errors[3], 'r_squared': r_squared, 'data_points': len(shear_rate), 'concentration': dataset['metadata']['concentration']}
                
                color = colors[i % len(colors)]; label = f"{dataset['metadata']['concentration']} g/L ({i+1})"
                axes[0, 0].loglog(shear_rate, viscosity, 'o', color=color, alpha=0.7, label=label)
                shear_fit = np.logspace(np.log10(min(shear_rate)), np.log10(max(shear_rate)), 100)
                axes[0, 0].loglog(shear_fit, self.cross_model(shear_fit, *popt), '-', color=color, alpha=0.8)
                
                all_conc.append(dataset['metadata']['concentration']); all_eta0.append(popt[0]); all_eta0_err.append(param_errors[0]); all_r2.append(r_squared); all_n.append(popt[3]); all_n_err.append(param_errors[3])
            except Exception as e:
                print(f"Fitting failed for {ds_id}: {e}")
        
        axes[0, 0].set(xlabel='Shear Rate [1/s]', ylabel='Viscosity [Pa·s]', title='Viscosity vs Shear Rate'); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].errorbar(all_conc, all_eta0, yerr=all_eta0_err, fmt='o', capsize=5, alpha=0.7); axes[0, 1].set(xlabel='Concentration [g/L]', ylabel='Zero-shear Viscosity η₀ [Pa·s]', title='η₀ vs Concentration', yscale='log'); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(all_conc, all_r2, 'o', alpha=0.7); axes[1, 0].set(xlabel='Concentration [g/L]', ylabel='R²', title='Fitting Quality', ylim=[0.9, 1.01]); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].errorbar(all_conc, all_n, yerr=all_n_err, fmt='o', capsize=5, alpha=0.7); axes[1, 1].set(xlabel='Concentration [g/L]', ylabel='Flow Behavior Index n', title='n vs Concentration'); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

    
# Ersetzen Sie auch diese Methode in Ihrer Klasse
   # Ersetzen Sie auch diese Methode
    def export_to_pdf_report(self, filename="rheology_report.pdf"):
        """Exports a detailed multi-page A4 PDF report, including fit ranges."""
        if not self.original_datasets: print("Error: No data loaded."); return
        print(f"Creating PDF report '{filename}'...")
        A4_SIZE_INCHES = (8.27, 11.69)
        
        with PdfPages(filename) as pdf:
            # --- SEITE 1: Titelseite mit Zusammenfassung ---
            fig_summary = plt.figure(figsize=A4_SIZE_INCHES)
            fig_summary.clf()
            fig_summary.text(0.5, 0.95, 'Rheological Analysis Summary', ha='center', size=16, weight='bold')
            model_info = (
                "Fit Function: Cross Model\n"
                r"Formula: $\eta = \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda \cdot \dot{\gamma})^n}$" + "\n"
                "Algorithm: Levenberg-Marquardt (scipy.optimize.curve_fit)"
            )
            fig_summary.text(0.1, 0.82, model_info, size=12, va='top')
    
            if self.fit_results:
                sorted_fit_results = sorted(self.fit_results.items(), key=lambda item: item[1]['concentration'])
                table_data = [[ds_id, f"{res['concentration']:.2f}", f"{res['eta_0']:.3f}", f"{res['n']:.3f}", f"{res['r_squared']:.4f}"] for ds_id, res in sorted_fit_results]
                columns = ['Dataset ID', 'Conc. (g/L)', 'η₀ (Pa·s)', 'n', 'R²']
                table = plt.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False); table.set_fontsize(8)
                table.scale(1, 1.5)
                plt.axis('off')
            pdf.savefig(fig_summary); plt.close(fig_summary)
    
            # --- FOLGESEITEN: Zweispaltiges Layout ---
            sorted_items = sorted(self.original_datasets.items(), key=lambda item: item[1]['metadata']['concentration'])
            sorted_ids = [item[0] for item in sorted_items]
            datasets_per_page = 3
            
            for i in range(0, len(sorted_ids), datasets_per_page):
                fig_page = plt.figure(figsize=A4_SIZE_INCHES, constrained_layout=True)
                gs = gridspec.GridSpec(datasets_per_page, 2, figure=fig_page, width_ratios=[3, 2], hspace=0.4)
                page_ids = sorted_ids[i:i + datasets_per_page]
                
                for j, ds_id in enumerate(page_ids):
                    # --- Linke Spalte: Diagramm ---
                    ax_plot = fig_page.add_subplot(gs[j, 0])
                    original_dataset = self.original_datasets[ds_id]
                    unfiltered_df = original_dataset['data']
                    ax_plot.loglog(unfiltered_df['shear_rate'], unfiltered_df['viscosity'], 'o', markersize=4, label='Original Data')
                    conc = original_dataset['metadata'].get('concentration', 'N/A')
                    ax_plot.set_title(f"ID: {ds_id}\nConc: {conc} g/L", fontsize=10)
                    if ds_id in self.fit_results:
                        res = self.fit_results[ds_id]
                        popt = [res['eta_0'], res['eta_inf'], res['lambda_c'], res['n']]
                        shear_fit = np.logspace(np.log10(unfiltered_df['shear_rate'].min()), np.log10(unfiltered_df['shear_rate'].max()), 100)
                        ax_plot.loglog(shear_fit, self.cross_model(shear_fit, *popt), '-', label="Fit", color='red')
                    ax_plot.set_xlabel("Scherrate [1/s]", fontsize=8); ax_plot.set_ylabel("Viskosität [Pa·s]", fontsize=8)
                    ax_plot.grid(True, which="both", ls="--", alpha=0.5); ax_plot.legend(fontsize=8)
    
                    # --- Rechte Spalte: Fit-Ergebnisse ---
                    ax_text = fig_page.add_subplot(gs[j, 1]); ax_text.axis('off')
                    
                    if ds_id in self.fit_results:
                        res = self.fit_results[ds_id]
                        # ✅ NEU: Fit-Grenzen holen und formatieren
                        fit_range = self.fit_ranges.get(ds_id, [-np.inf, np.inf])
                        range_min = f"{fit_range[0]:.2f}" if np.isfinite(fit_range[0]) else "min"
                        range_max = f"{fit_range[1]:.2f}" if np.isfinite(fit_range[1]) else "max"
                        range_str = f"Fit Range [1/s]: {range_min} - {range_max}"
    
                        fit_text = (
                            r"$\bf{Fit\ Results}$" + "\n"
                            f"$R^2$ = {res['r_squared']:.4f}\n"
                            f"Data Points = {res['data_points']}\n"
                            f"{range_str}\n\n" # ✅ HIER EINGEFÜGT
                            r"$\bf{Parameters}$" + "\n"
                            fr"$\eta_0$ = {res['eta_0']:.3f} $\pm$ {res['eta_0_error']:.3f} Pa·s" + "\n"
                            fr"$\eta_\infty$ = {res['eta_inf']:.3f} $\pm$ {res['eta_inf_error']:.3f} Pa·s" + "\n"
                            fr"$\lambda$ = {res['lambda_c']:.3f} $\pm$ {res['lambda_c_error']:.3f} s" + "\n"
                            fr"$n$ = {res['n']:.3f} $\pm$ {res['n_error']:.3f}"
                        )
                        ax_text.text(0.05, 0.95, fit_text, transform=ax_text.transAxes, fontsize=9,
                                     verticalalignment='top', horizontalalignment='left',
                                     bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))
                    else:
                        ax_text.text(0.5, 0.5, "No fit performed for this dataset.", ha='center', va='center', fontsize=9)
                
                pdf.savefig(fig_page); plt.close(fig_page)
        
        print(f"✅ PDF report created successfully.")

  
    # Ersetzen Sie diese Methode in Ihrer Klasse
    def export_results(self, filename='rheological_analysis_results.txt'):
        """Exports a summary table and detailed fitting results, including fit ranges."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Rheological Analysis Results\n")
            f.write("="*80 + "\n")
            f.write("Fit Function: Cross Model\n")
            f.write("Formula: η = η_inf + (η_0 - η_inf) / (1 + (λ * γ̇)^n)\n")
            f.write("Fitting Algorithm: Levenberg-Marquardt (via scipy.optimize.curve_fit)\n")
            f.write("="*80 + "\n\n")
    
            # Sortiere die Ergebnisse nach Konzentration
            sorted_fit_results = sorted(self.fit_results.items(), key=lambda item: item[1]['concentration'])
    
            # --- Zusammenfassende Tabelle ---
            f.write("--- Summary Table ---\n")
            header = f"{'Dataset ID':<40} {'Conc. [g/L]':<15} {'η₀ [Pa·s]':<15} {'n':<10} {'R²':<10}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")
            for ds_id, res in sorted_fit_results:
                f.write(
                    f"{ds_id:<40} "
                    f"{res['concentration']:<15.2f} "
                    f"{res['eta_0']:<15.3f} "
                    f"{res['n']:<10.3f} "
                    f"{res['r_squared']:<10.4f}\n"
                )
            f.write("\n" + "="*80 + "\n\n")
    
            # --- Detaillierte Einzelergebnisse ---
            f.write("--- Detailed Fit Results ---\n\n")
            for ds_id, res in sorted_fit_results:
                f.write(f"--- Dataset ID: {ds_id} ---\n")
                f.write(f"  Concentration: {res['concentration']} g/L\n")
                
                # ✅ NEU: Fit-Grenzen hinzufügen
                fit_range = self.fit_ranges.get(ds_id, [-np.inf, np.inf])
                range_min = f"{fit_range[0]:.2f}" if np.isfinite(fit_range[0]) else "min"
                range_max = f"{fit_range[1]:.2f}" if np.isfinite(fit_range[1]) else "max"
                f.write(f"  Fit Range [1/s]: {range_min} to {range_max}\n")
                
                f.write(f"  Fit Quality (R²): {res['r_squared']:.4f}\n")
                f.write(f"  Data Points Used: {res['data_points']}\n")
                f.write("  Parameters:\n")
                f.write(f"    η₀ (Zero-shear viscosity) : {res['eta_0']:.6f} ± {res['eta_0_error']:.6f} Pa·s\n")
                f.write(f"    η∞ (Inf-shear viscosity)  : {res['eta_inf']:.6f} ± {res['eta_inf_error']:.6f} Pa·s\n")
                f.write(f"    λ (Relaxation time)       : {res['lambda_c']:.6f} ± {res['lambda_c_error']:.6f} s\n")
                f.write(f"    n (Flow behavior index)   : {res['n']:.4f} ± {res['n_error']:.4f}\n\n")
                
        print(f"Detailed results exported to {filename}")

#!/usr/bin/env python3
"""
nmap_gui_scan.py

Simple Tkinter GUI wrapper for nmap that:
 - Asks the user for network/hosts to scan
 - Lets user choose ports (all, common, custom)
 - Provides options (SYN scan, -A, timing)
 - Runs nmap in background thread, shows live output and saves results (-oA)
 - Requires explicit agreement to legal warning before running

Usage:
    python3 nmap_gui_scan.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import subprocess
import shlex
import shutil
import datetime
import os
import sys

# ---------- helper functions ----------
def find_nmap():
    return shutil.which("nmap")

def sanitize_arg(arg):
    # Basic sanity: strip; let shlex.quote handle shell safety when building command
    return arg.strip()

def build_nmap_command(target, port_option, custom_ports, syn_scan, enable_A, timing, extra_args, output_basename):
    # Build list of args (not a shell string)
    cmd = []
    cmd.append("nmap")

    # Timing template
    if timing and timing != "T3":
        cmd.append(f"-{timing}")

    # Scan type
    if syn_scan:
        cmd.append("-sS")
    else:
        cmd.append("-sT")  # TCP connect

    # Advanced
    if enable_A:
        cmd.append("-A")

    # Ports
    if port_option == "all":
        cmd.append("-p-")
    elif port_option == "common":
        # nmap default scan covers common ports; don't add -p so it uses defaults
        pass
    elif port_option == "custom":
        p = custom_ports.strip()
        if p:
            cmd.extend(["-p", p])

    # Additional arguments user may want
    if extra_args:
        # split safely
        extra_parts = shlex.split(extra_args)
        cmd.extend(extra_parts)

    # Output base name (nmap will append .nmap .xml .gnmap with -oA)
    if output_basename:
        cmd.extend(["-oA", output_basename])

    # Finally the target(s)
    cmd.append(target)

    return cmd

# ---------- GUI ----------
class NmapGUI:
    def __init__(self, root):
        self.root = root
        root.title("Nmap Simple GUI Scanner")
        root.geometry("850x640")

        # Main frame
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Warning / permission
        self.agree_var = tk.BooleanVar(value=False)
        warn = ("WARNING: Only scan machines you own or have permission to scan.\n"
                "Unauthorized scanning may be illegal.")
        ttk.Label(frm, text=warn, foreground="darkred").pack(anchor="w", pady=(0,6))
        ttk.Checkbutton(frm, text="I have permission to scan the target(s)", variable=self.agree_var).pack(anchor="w", pady=(0,8))

        # Target entry
        tgt_frame = ttk.Labelframe(frm, text="Target(s) / Network (examples: 192.168.1.1, 192.168.1.0/24, host.example.com, 10.0.0.1-50)")
        tgt_frame.pack(fill=tk.X, pady=4)
        self.target_var = tk.StringVar(value="192.168.1.0/24")
        ttk.Entry(tgt_frame, textvariable=self.target_var).pack(fill=tk.X, padx=6, pady=6)

        # Ports options
        ports_frame = ttk.Labelframe(frm, text="Port selection")
        ports_frame.pack(fill=tk.X, pady=4)
        self.port_choice = tk.StringVar(value="common")
        ttk.Radiobutton(ports_frame, text="Common ports (nmap default)", variable=self.port_choice, value="common").pack(anchor="w", padx=6, pady=2)
        ttk.Radiobutton(ports_frame, text="All ports (0-65535) - slower", variable=self.port_choice, value="all").pack(anchor="w", padx=6, pady=2)
        custom_rb = ttk.Radiobutton(ports_frame, text="Custom ports / ranges (e.g. 22,80,443,8000-8100)", variable=self.port_choice, value="custom")
        custom_rb.pack(anchor="w", padx=6, pady=(2,0))
        self.custom_ports_var = tk.StringVar()
        ttk.Entry(ports_frame, textvariable=self.custom_ports_var).pack(fill=tk.X, padx=28, pady=6)

        # Options frame
        opts = ttk.Labelframe(frm, text="Scan options")
        opts.pack(fill=tk.X, pady=4)
        self.syn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Use SYN scan (requires root/privileged)", variable=self.syn_var).pack(anchor="w", padx=6, pady=2)
        self.adv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Enable -A (OS/version/script/traceroute) (slower / intrusive)", variable=self.adv_var).pack(anchor="w", padx=6, pady=2)
        ttk.Label(opts, text="Timing template:").pack(anchor="w", padx=6, pady=(6,0))
        self.timing_var = tk.StringVar(value="T3")
        ttk.Combobox(opts, textvariable=self.timing_var, values=["T0","T1","T2","T3","T4","T5"], width=6).pack(anchor="w", padx=6, pady=(0,6))

        ttk.Label(opts, text="Extra nmap args (advanced users):").pack(anchor="w", padx=6, pady=(6,0))
        self.extra_args_var = tk.StringVar()
        ttk.Entry(opts, textvariable=self.extra_args_var).pack(fill=tk.X, padx=6, pady=(0,6))

        # Output controls
        out_frame = ttk.Frame(frm)
        out_frame.pack(fill=tk.X, pady=4)
        ttk.Label(out_frame, text="Output basename (files will be saved as basename.nmap/.xml/.gnmap):").pack(anchor="w", padx=6)
        self.basename_var = tk.StringVar(value=f"nmap_scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ttk.Entry(out_frame, textvariable=self.basename_var).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(out_frame, text="Choose output folder...", command=self.choose_output_folder).pack(anchor="e", padx=6)
        self.output_folder = os.getcwd()
        ttk.Label(out_frame, textvariable=tk.StringVar(value=f"Output folder: {self.output_folder}")).pack(anchor="w", padx=6, pady=(2,4))
        self.output_folder_label = out_frame.winfo_children()[-1]

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill=tk.X, pady=8)
        self.start_btn = ttk.Button(btn_frame, text="Start Scan", command=self.start_scan)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_requested = threading.Event()
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        # Real-time output pane
        log_frame = ttk.Labelframe(frm, text="Scan output")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.configure(state=tk.DISABLED)

        # Status at bottom
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

        # Ensure nmap exists
        if not find_nmap():
            messagebox.showerror("nmap not found", "nmap executable not found in PATH. Please install nmap and ensure it's in your PATH.")
            self.start_btn.configure(state=tk.DISABLED)

    def choose_output_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder)
        if folder:
            self.output_folder = folder
            self.output_folder_label.configure(text=f"Output folder: {self.output_folder}")

    def append_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def request_stop(self):
        self.stop_requested.set()
        self.append_log("\nStop requested. Attempting to terminate nmap...\n")
        self.status_var.set("Stopping...")

    def start_scan(self):
        if not self.agree_var.get():
            messagebox.showwarning("Permission required", "You must confirm you have permission to scan the targets.")
            return

        target = sanitize_arg(self.target_var.get())
        if not target:
            messagebox.showwarning("Missing target", "Enter a target or network to scan.")
            return

        port_option = self.port_choice.get()
        custom_ports = self.custom_ports_var.get()
        syn_scan = self.syn_var.get()
        enable_A = self.adv_var.get()
        timing = self.timing_var.get()
        extra_args = self.extra_args_var.get()
        basename = sanitize_arg(self.basename_var.get()) or f"nmap_scan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_basepath = os.path.join(self.output_folder, basename)

        cmd = build_nmap_command(target=target,
                                 port_option=port_option,
                                 custom_ports=custom_ports,
                                 syn_scan=syn_scan,
                                 enable_A=enable_A,
                                 timing=timing,
                                 extra_args=extra_args,
                                 output_basename=out_basepath)

        # Show command for transparency
        cmd_display = " ".join(shlex.quote(x) for x in cmd)
        if not messagebox.askokcancel("Confirm scan", f"The following nmap command will be run:\n\n{cmd_display}\n\nProceed?"):
            return

        # Disable start button, enable stop
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.stop_requested.clear()
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self.status_var.set("Running scan...")
        self.append_log(f"Starting: {cmd_display}\n\n")

        # Run in background thread
        t = threading.Thread(target=self.run_nmap, args=(cmd, out_basepath), daemon=True)
        t.start()

    def run_nmap(self, cmd, out_basepath):
        try:
            # Launch subprocess, capture stdout/stderr line by line
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            # Store for potential termination
            self._proc = proc

            for line in proc.stdout:
                if self.stop_requested.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                self.append_log(line)

            proc.wait(timeout=10)
            rc = proc.returncode
            if self.stop_requested.is_set():
                self.append_log("\nScan stopped by user.\n")
                self.status_var.set("Stopped")
            else:
                self.append_log(f"\nScan finished (return code {rc}). Output files saved with basename: {out_basepath}\n")
                self.status_var.set("Finished")
        except FileNotFoundError:
            self.append_log("\nError: nmap not found. Ensure nmap is installed and in PATH.\n")
            self.status_var.set("Error")
        except Exception as e:
            self.append_log(f"\nError running nmap: {e}\n")
            self.status_var.set("Error")
        finally:
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self._proc = None
            self.stop_requested.clear()


# ---------- run ----------
def main():
    if sys.version_info < (3,6):
        print("Python 3.6+ is required.")
        sys.exit(1)

    root = tk.Tk()
    gui = NmapGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
from Optimize_PIP_PIP import Optimize_PIP_PIP
from Optimize_PIP_MOL import Optimize_PIP_MOL
import matplotlib.dates as mdates
from tkintermapview import TkinterMapView 
import os



def open_map():
    map_window = tk.Toplevel(root)
    map_window.title("Select Location on Map")

    global map_view
    map_view = TkinterMapView(map_window, width=800, height=600)
    map_view.pack(fill="both", expand=True)

    # Center map and set default zoom
    map_view.set_position(35.0, -118.0)  # Example coordinates
    map_view.set_zoom(5)

    # Define the function to populate entries
    def populate_entries_with_coordinates(coords):
        lat, lon = coords
        latitude_entry.delete(0, tk.END)
        longitude_entry.delete(0, tk.END)
        latitude_entry.insert(0, f"{lat:.6f}")
        longitude_entry.insert(0, f"{lon:.6f}")
        # Set a marker at the selected location
        map_view.set_marker(lat, lon)

    # Add custom command to the right-click menu
    map_view.add_right_click_menu_command(label="Set as Selected Location",
                                          command=populate_entries_with_coordinates,
                                          pass_coords=True)
################################################################
def load_and_plot_ain_aen(files,pref):
################################################################
    try:
        # Load the required files        
        dates_file = os.path.join(pref, "dates.txt")
        dates = pd.read_csv(dates_file, header=None, parse_dates=True)[0]        
        for ttl, flnm in files.items():
            print("Key:", ttl)
            print("Value:", flnm)
            ain = pd.read_csv(os.path.join(pref, flnm[0]), header=None)[0]
            af = pd.read_csv(os.path.join(pref, flnm[1]), header=None)[0]
            create_interactive_plot(dates, ain, af, "Before Control Measures", "After Control Measures", ttl)
           
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def create_interactive_plot(dates, data1, data2, label1, label2, title):
    # Ensure dates are in the correct datetime format
    dates = pd.to_datetime(dates, errors='coerce')

    # Create a fully interactive plot using matplotlib's default interactive window
    fig, ax = plt.subplots()
    ax.plot(dates, data1,'.-b', label=label1,linewidth=1)
    ax.plot(dates, data2,'.-r', label=label2,linewidth=1)
    ax.set_xlabel("Dates")
    ax.set_ylabel("Values")
    ax.set_title(title)
    ax.legend()

    # Format x-axis to show a maximum of 10 dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()  # Rotate date labels for better readability

    # Show the interactive plot in a separate matplotlib window
    plt.show()
################################################################
def show_dates_from_files(dtfiles,pref):
    
    for ttl, flnm in dtfiles.items():
            print("Key:", ttl)
            print("Value:", flnm)

            try:
                with open(os.path.join(pref, flnm), "r") as f:
                    dates = f.readlines()
    
                # Create a new window for the file content
                date_window = tk.Toplevel(root)
                date_window.title(ttl)
    
                text_widget = tk.Text(date_window, wrap=tk.WORD, width=50, height=20)
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
                scrollbar = tk.Scrollbar(date_window, command=text_widget.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                text_widget.config(yscrollcommand=scrollbar.set)
    
                # Insert dates into the text widget
                for date in dates:
                    text_widget.insert(tk.END, date)
    
            except FileNotFoundError:
                messagebox.showerror("Error", f"File {flnm} not found.")






#####################################################################################################################################

def Culex_pipiens_pip_window(): 
    
################################################################
    def run_optimization():
        try:

            # Run the Python optimization function
            result = Optimize_PIP_PIP(
                longitude=longitude_entry.get(),
                latitude=latitude_entry.get(),
                len_ins=entries["len_ins"].get(),
                len_lr=entries["len_lr"].get(),
                len_cr=entries["len_cr"].get(),
                numtrls=entries["numtrls"].get(),
                ls_ef=entries["ls_ef"].get(),
                numtris=entries["numtris"].get(),
                is_ef=entries["is_ef"].get(),
                numtris_di=entries["numtris_di"].get(),
                is_ef_di=entries["is_ef_di"].get(),
                numtrcl=entries["numtrcl"].get(),
                cl_ef=entries["cl_ef"].get(),
                adtemp=entries["adtemp"].get(),
                adpre=entries["adpre"].get()
            )

            # Display the result
            #plot_result(result)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create a new window for input
    input_window = tk.Toplevel(root)
    input_window.title("Culex_pipiens_pip_window")
    
   
    # Variable labels and entries
    variables = [
        ("Duration of insecticide's effectiveness", "len_ins", "2"),
        ("Number of insecticide application periods", "numtris", "2"),
        ("Efficiency of insecticide", "is_ef", "0.2"),
        ("Number of insecticide application periods for di", "numtris_di", "1"),
        ("Efficiency of insecticide for di", "is_ef_di", "0.2"),
        ("Duration of larvicide's effectiveness", "len_lr", "20"),
        ("Number of larvicide application periods", "numtrls", "2"),
        ("Efficiency of larvicide", "ls_ef", "0.8"),
        ("Duration of habitat elimination", "len_cr", "40"),
        ("Number of habitat elimination periods", "numtrcl", "1"),
        ("Efficiency of habitat elimination", "cl_ef", "0.5"),
        ("Temperature file Path", "adtemp", "temp_pastyearsav_py.mat"),
        ("Precipitation file Path", "adpre", "pre_pastyearsav_py.mat"),
    ]

    entries = {}

    for i, (label_text, var_name, default_value) in enumerate(variables):
        label = tk.Label(input_window, text=label_text)
        label.grid(row=i+2, column=0, sticky="e")

        entry = tk.Entry(input_window, width=40)
        entry.insert(0, default_value)
        entry.grid(row=i+2, column=1)
        
        entries[var_name] = entry
   
    tk.Label(input_window, text="Longitude").grid(row=0, column=0, sticky="e")
    global longitude_entry
    longitude_entry = tk.Entry(input_window, width=40)
    longitude_entry.insert(0, "-74.28")
    longitude_entry.grid(row=0, column=1)
    
    tk.Label(input_window, text="Latitude").grid(row=1, column=0, sticky="e")
    global latitude_entry
    latitude_entry = tk.Entry(input_window, width=40)
    latitude_entry.insert(0, "40.8")
    latitude_entry.grid(row=1, column=1)
    tk.Button(input_window, text="Open Map", command=open_map).grid(row=1, column=2, columnspan=2, pady=10)
    

    # Add browse buttons for file paths
    def browse_file(entry):
        filepath = filedialog.askopenfilename()
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)

    for var_name in ["adtemp", "adpre"]:
        button = tk.Button(input_window, text="Browse", command=lambda e=entries[var_name]: browse_file(e))
        button.grid(row=[v[1] for v in variables].index(var_name)+2, column=2)

    # Add run button
    run_button = tk.Button(input_window, text="Run Optimization", command=run_optimization)
    run_button.grid(row=len(variables)+4, column=0)
    
    
    pref= "pipien_pip"
    
    piprisk={"Risk" :["riskin.txt", "riskf.txt"]}
    rin_ren_button = tk.Button(input_window, text="Plot Risk", command=lambda: load_and_plot_ain_aen(piprisk,pref))
    rin_ren_button.grid(row=len(variables)+6, column=0)
    
    
    dipops={"Mosquito Population" :["ain.txt", "af.txt"], "Mosquito Population in diapause" :["a_diin.txt", "a_dif.txt"]}
    ain_aen_button = tk.Button(input_window, text="Plot Mosquito Population", command=lambda: load_and_plot_ain_aen(dipops,pref))
    ain_aen_button.grid(row=len(variables)+8, column=0)

    didats={"Otimal dates for applying larvicide" :"datels.txt", 
             "Optimal dates for applying insecticide" : "dateis.txt",
             "Optimal dates for applying insecticide_diapause" : "dateis_di.txt",
             "Optimal dates for habitat elimination": "datetau.txt"}
    show_dates_button = tk.Button(input_window, text="Show Optimal Dates", command=lambda: show_dates_from_files(didats,pref))
    show_dates_button.grid(row=len(variables)+10, column=0)
    
def Culex_pipiens_mol_window():
    
    def run_optimization():
        try:
            # Collect inputs
            
            # Run the Python optimization function
            Optimize_PIP_MOL(
               longitude=longitude_entry.get() ,
               latitude=latitude_entry.get() , 
               len_ins=entries["len_ins"].get() ,   
               len_lr=entries["len_lr"].get()  ,     
               len_cr=entries["len_cr"].get()   ,      
               numtrls=entries["numtrls"].get()   , 
               ls_ef=entries["ls_ef"].get()   ,   
               numtris=entries["numtris"].get()  ,    
               is_ef=entries["is_ef"].get()  , 
               numtrcl=entries["numtrcl"].get()  ,   
               cl_ef=entries["cl_ef"].get()  ,  
               numtrls_mh=entries["numtrls_mh"].get() ,  
               ls_ef_mh=entries["ls_ef_mh"].get() ,     
               numtris_mh=entries["numtris_mh"].get()  ,   
               is_ef_mh=entries["is_ef_mh"].get()  , 
               numtrcl_mh=entries["numtrcl_mh"].get()  ,   
               cl_ef_mh=entries["cl_ef_mh"].get() ,  
               adtemp=entries["adtemp"].get(),
               adpre=entries["adpre"].get() 
            )

            # Display the result
            #plot_result(result)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create a new window for input
    input_window = tk.Toplevel(root)
    input_window.title("Culex_pipiens_mol_window")
    
   
    # Variable labels and entries
    variables = [
        ("Duration of insecticide's effectiveness", "len_ins", "2"),
        ("Number of insecticide application periods", "numtris", "2"),
        ("Efficiency of insecticide", "is_ef", "0.2"),
        ("Number of insecticide application periods for mh", "numtris_mh", "1"),
        ("Efficiency of insecticide for mh", "is_ef_mh", "0.2"),
        ("Duration of larvicide's effectiveness", "len_lr", "20"),
        ("Number of larvicide application periods", "numtrls", "2"),
        ("Efficiency of larvicide", "ls_ef", "0.8"),
        ("Number of larvicide application periods for mh", "numtrls_mh", "1"),
        ("Efficiency of larvicide for mh", "ls_ef_mh", "0.8"),
        ("Duration of habitat elimination", "len_cr", "40"),
        ("Number of habitat elimination periods", "numtrcl", "1"),
        ("Efficiency of habitat elimination", "cl_ef", "0.5"),
        ("Number of habitat elimination periods for mh", "numtrcl_mh", "1"),
        ("Efficiency of habitat elimination for mh", "cl_ef_mh", "0.5"),
        ("Temperature file Path", "adtemp", "temp_pastyearsav_py.mat"),
        ("Precipitation file Path", "adpre", "pre_pastyearsav_py.mat"),
    ]

    entries = {}

    for i, (label_text, var_name, default_value) in enumerate(variables):
        label = tk.Label(input_window, text=label_text)
        label.grid(row=i+2, column=0, sticky="e")

        entry = tk.Entry(input_window, width=40)
        entry.insert(0, default_value)
        entry.grid(row=i+2, column=1)
        
        entries[var_name] = entry
   
    tk.Label(input_window, text="Longitude").grid(row=0, column=0, sticky="e")
    global longitude_entry
    longitude_entry = tk.Entry(input_window, width=40)
    longitude_entry.insert(0, "-74.28")
    longitude_entry.grid(row=0, column=1)
    
    tk.Label(input_window, text="Latitude").grid(row=1, column=0, sticky="e")
    global latitude_entry
    latitude_entry = tk.Entry(input_window, width=40)
    latitude_entry.insert(0, "40.8")
    latitude_entry.grid(row=1, column=1)
    tk.Button(input_window, text="Open Map", command=open_map).grid(row=1, column=2, columnspan=2, pady=10)
    


    # Add browse buttons for file paths
    def browse_file(entry):
        filepath = filedialog.askopenfilename()
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)

    for var_name in ["adtemp", "adpre"]:
        button = tk.Button(input_window, text="Browse", command=lambda e=entries[var_name]: browse_file(e))
        button.grid(row=[v[1] for v in variables].index(var_name)+2, column=3)

    # Add run button
    run_button = tk.Button(input_window, text="Run Optimization", command=run_optimization)
    run_button.grid(row=len(variables)+4, column=0)
    
    pref= "pipien_mol"
    molrisk={"Risk" :["riskin.txt", "riskf.txt"]}
    rin_ren_button = tk.Button(input_window, text="Plot Risk", command=lambda: load_and_plot_ain_aen(molrisk,pref))
    rin_ren_button.grid(row=len(variables)+6, column=0)
    
    
    molpops={"Mosquito Population" :["ain.txt", "af.txt"], "Mosquito Population in MH" :["a_mhin.txt", "a_mhf.txt"]}
    ain_aen_button = tk.Button(input_window, text="Plot Mosquito Population", command=lambda: load_and_plot_ain_aen(molpops,pref))
    ain_aen_button.grid(row=len(variables)+8, column=0)
      
    moldats={"Otimal dates for applying larvicide" :"datels.txt", 
             "Otimal dates for applying larvicide MH" :"datels_mh.txt",
             "Optimal dates for applying insecticide" : "dateis.txt",
             "Optimal dates for applying insecticide MH" : "dateis_mh.txt",
             "Optimal dates for habitat elimination": "datetau.txt",
             "Optimal dates for habitat elimination MH": "datetau_mh.txt"}
    show_dates_button = tk.Button(input_window, text="Show Optimal Dates", command=lambda: show_dates_from_files(moldats,pref))
    show_dates_button.grid(row=len(variables)+10, column=0)    
     
    
###################################################################################################################################    
root = tk.Tk()
root.title("Interactive Plotting GUI")



Culex_pipiens_pip_button = tk.Button(root, text="Culex pipiens (pipien) mosquito", command=Culex_pipiens_pip_window)
Culex_pipiens_pip_button.pack(pady=10)

Culex_pipiens_mol_button = tk.Button(root, text="Culex pipiens (molestus) mosquito", command=Culex_pipiens_mol_window)
Culex_pipiens_mol_button.pack(pady=10)

# Run the application
root.mainloop()

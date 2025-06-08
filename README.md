# Ancillary Service Calculator (ASC)

An open-source Python based tool, designed for network service providers to estimate the frequency regulation and voltage support services available from the distribution networks to the upstream networks (e.g., transmission networks).
This tool can be used in tandem with DigSILENT PowerFactory. A short technical paper about this tool is currently under review in SoftwareX.

## Main features

- The tool utilized a three-step time aggregation method to obtain representative profiles of loads, solar generations, wind generations, and charging characteristics of electric vehicles based on their corresponding historical time series.
- It is also capable of assessing the security measures such as under/over voltage violations and overloading of the network by interfacing with backbone power system analysis software (in this case DigSILENT PowerFactory).
- It helps users to estimate the amount of frequency regulation and reactive power ancillary supports from a distribution network to the upstream network using time aggregated profiles of the demand, DERs, and EV charging stations.

## Installation & How to run it
- The source code is developed in Python 3.11 (details of the libraries used can be view in the Python code file). The GUI is developed by PyQT5.
- The tool is portable, no installation is required. Currently, the tool can only be used on Windows.

## An overview of the user interface and how to use the ASC
**Stage 1**:

At this stage, users can simply put the input Excel file into the same folder which contains the ASC for easy browsing, or they can click on "Browse" and find their input Excel file.
Note that the input Excel file should follow the simple structure as shown in the next figure. In our case study, the profiles (time series) has a 30-min resolution. This can be easily modified in the source code according to your needs.

The steps to perform time aggregation are: Browse for input Excel file -> Plot (optional step) -> Calculate the optimal number of cluster -> Generate representative profiles -> Save (optional, use this if you want to export the representative profiles to Excel files)

![Image](https://github.com/user-attachments/assets/b663626a-e865-4a01-8813-24805b6c2992)

Excel file format 

![Image](https://github.com/user-attachments/assets/377e45a3-8abd-4fa0-8c10-d9c98e1b2336)

**Stage 2**:

Firstly, make sure that you follow the steps in Stage 1, since in this stage, the tool uses the representative profiles generated in Stage 1 (the representative profiles are saved internally when you click on "Generate representative profiles").

Secondly, provide the link to the PowerFactory's Python API, and the name of the **case study** associated with your model file (this name is obtained when you activate your pfd file inside DigSILENT PowerFactory). The "Security check" is optional, you can directly calculate the ancillary services by providing your desired parameters, and click on "Calculate".



![Image](https://github.com/user-attachments/assets/5b06e186-f4cf-4e9e-89f1-181bd8aa9ecf)


## Main developer 
- Any issues or suggestions please feel free to contact [Tan Nhat Pham](https://github.com/nhattan214)

## Further development
I am working on newer versions that allow the ASC to be used with pandapower, or MATPOWER.

## License info
This project has been developed in the Center for New Energy Transition Research (CfNETR), Federation University Australia, and it is licensed under the terms of the MIT License.

## Acknowledgements
This research is part of a program of fourteen projects, which have variously been funded and supported under C4NET’s Enhanced System Planning (ESP) collaborative research project [Work Package 3.11 & 3.12](https://c4net.com.au/projects/enhanced-system-planning-project/). While conducted independently, the authors acknowledge the funding and in-kind support of C4NET, its members and supporters, and its ESP project partners.

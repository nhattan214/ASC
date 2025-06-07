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

## Main developer 


## License info
This project has been developed in the Center for New Energy Transition Research (CfNETR), Federation University Australia, and it is licensed under the terms of the MIT License.

## Acknowledgements
This research is part of a program of fourteen projects, which have variously been funded and supported under C4NET’s Enhanced System Planning (ESP) collaborative research project [Work Package 3.11 & 3.12](https://c4net.com.au/projects/enhanced-system-planning-project/). While conducted independently, the authors acknowledge the funding and in-kind support of C4NET, its members and supporters, and its ESP project partners.

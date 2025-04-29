# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:29:47 2025

@author: Tan Pham
"""


import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication,QMainWindow, QMessageBox, QFileDialog, QDesktopWidget
from PyQt5.uic import loadUi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import random

class Solver(QMainWindow):
    
    def __init__(self):
        super(Solver,self).__init__()
        loadUi('Gui_mk1.ui',self)
        self.browse_button_stage_1.clicked.connect(self.browse_file_stage_1)
        self.contact_info_button.clicked.connect(self.show_contact_info)
        self.plot_button_stage_1.clicked.connect(self.plot_stage_1)
        self.gen_rep_profile_button_si.clicked.connect(self.gen_rep_profile)
        self.generate_rep_profile_manual_input.clicked.connect(self.gen_rep_profile_manual)
        self.save_rep_profile_button.clicked.connect(self.save_rep_profile)
        self.browse_stage_2_button.clicked.connect(self.browse_file_stage_2)
        self.security_check_button.clicked.connect(self.network_security_check_v2)
        self.calculate_button_stage_2.clicked.connect(self.ancillary_calculation)
        
        
        self.dr_up_reserve.setText('100')
        self.dr_down_reserve.setText('100')
        self.pv_up_reserve.setText('100')
        self.pv_down_reserve.setText('100')
        self.wind_up_reserve.setText('100')
        self.wind_down_reserve.setText('100')
        self.ev_up_reserve.setText('100')
        self.ev_down_reserve.setText('100')
        self.ess_up_reserve.setText('100')
        self.ess_down_reserve.setText('100')
        self.total_ess_capacity.setText('5')
        self.percentage_of_willingness_ess.setText('50')
        
        
        self.python_folder_address=-1
        self.file_address=-1
        self.Load_profile = -1 
        self.PV_profile=-1
        self.Wind_profile=-1
        self.EV_profile=-1
        self.opt_num_cluster=-1
        self.flag_save=0
        self.flag_file_ready=0
        
    def browse_file_stage_1(self):
        file_name=QFileDialog.getOpenFileName(self, 'Open File', '*.xlsx')
        # print(file_name)
        self.file_address_stage_1.setText(file_name[0])
        self.file_address = file_name[0]
        
        self.flag_file_ready=1
        return self.file_address, self.flag_file_ready
    
    def browse_file_stage_2(self):
        file_name=QFileDialog.getExistingDirectory(self, 'Select an awesome directory')
        self.link_to_python_folder.setText(file_name)
        self.python_folder_address = file_name
        # print(file_name)
        # sys.path.append(file_name)
        return self.python_folder_address
    
    def network_security_check(self):
        sys.path.append(self.python_folder_address)
        import powerfactory as pf
    
        # try: 
        #    app1=pf.GetApplicationExt()
        # except pf.ExitError as error:
        #     # print(error)
        #     # print('error.code = %d' % error.code)
        #     msg = QMessageBox()
        #     msg.setWindowTitle('Error')
        #     msg.setText('Cannot connect to PowerFactory, please check your Internet connection, your license, or close your PowerFactory before using the tool')
        #     msg.setIcon(QMessageBox.Critical)
        #     x=msg.exec_()
        # else:
        app = pf.GetApplication()

        app.ClearOutputWindow()

        case_name=self.stage_2_case_name.text()
        
        if len(case_name)==0:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Please enter the case study name')
            msg.setIcon(QMessageBox.Warning)
            x=msg.exec_()
            print('nocase')
        else:
            print(case_name)
            
            app.ActivateProject(case_name)
    
            prj = app.GetActiveProject()
    
            filename = prj.GetAttribute("loc_name")
            loads = app.GetCalcRelevantObjects('*.ElmLod')
            load_dict = {}
            for i,load in enumerate(loads):
              # print(i, load.loc_name)
              load_dict[load.loc_name] = load
              
            #get all buses
            buses = app.GetCalcRelevantObjects('*.ElmTerm')
            bus_dict = {}
            for bus in buses:
              bus_dict[bus.loc_name] = bus
    
            #get all lines
            lines=app.GetCalcRelevantObjects('*.ElmLne')
            line_dict={}
            for line in lines:
              line_dict[line.loc_name] = line
    
            #retrieve load-flow object
            ldf = app.GetFromStudyCase("ComLdf")
    
            ldf.iopt_net = 0
             #execute load flow
            ldf.Execute()
    
            all_voltage=[]
            all_busname=[]
            all_linename=[]
            all_line_loading=[]
    
            all_load_bus=[]
            all_load_values=[]
    
    
            violated_voltage_bus_name=[]
            violated_voltage_value=[]
            violated_loading_line_name=[]
            violated_loading_value=[]

            for n,bus_key in enumerate(bus_dict.keys()):
              
              
              bus_voltage=bus_dict[bus_key].GetAttribute("m:u1")
              all_voltage.append(bus_voltage)
              all_busname.append(bus_key)
              if bus_voltage<0.94 or bus_voltage>1.05:
                  violated_voltage_bus_name.append(bus_key)
                  violated_voltage_value.append(bus_voltage)
              
            for n, line_key in enumerate(line_dict.keys()):
                line_loading=line_dict[line_key].GetAttribute("c:loading")
                all_linename.append(line_key)
                all_line_loading.append(line_loading)
                if line_loading>100:
                    violated_loading_line_name.append(line_key)
                    violated_loading_value.append(line_loading)

            for n, load_key in enumerate(load_dict.keys()):
                load_value=load_dict[load_key].GetAttribute("m:P:bus1")
                all_load_bus.append(load_key)
                all_load_values.append(load_value)


            violated_voltage_dataframe={'Bus name': violated_voltage_bus_name, 'Voltage value':violated_voltage_value}
            violated_voltage_dataframe=pd.DataFrame(violated_voltage_dataframe)

            violated_loading_dataframe={'Line name': violated_loading_line_name, 'Loading':violated_loading_value}
            violated_loading_dataframe=pd.DataFrame(violated_loading_dataframe)
              
            if len(violated_voltage_bus_name) ==0 and len(violated_loading_line_name) ==0:
                msg = QMessageBox()
                msg.setWindowTitle('Info')
                msg.setText('No security violation')
                msg.setIcon(QMessageBox.Information)
                x=msg.exec_()
            else:
                msg = QMessageBox()
                msg.setWindowTitle('Info')
                msg.setText('Security violation detected!!'+' There are ' + str(len(violated_voltage_bus_name))+' overloaded bus and '+str(len(violated_loading_line_name))+' overloaded line. See the excel file for further information.')
                msg.setIcon(QMessageBox.Warning)
                x=msg.exec_()
                with pd.ExcelWriter("Violation.xlsx") as writer:
           
                    # use to_excel function and specify the sheet_name and index 
                    # to store the dataframe in specified sheet
                    violated_voltage_dataframe.to_excel(writer, sheet_name="Violated voltage", index=False)
                    violated_loading_dataframe.to_excel(writer, sheet_name="Violated line loading", index=False)
    
    def network_security_check_v2(self):
        sys.path.append(self.python_folder_address)
        import powerfactory as pf
    
      
        app = pf.GetApplication()

        app.ClearOutputWindow()

        case_name=self.stage_2_case_name.text()
        
        if len(case_name)==0:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Please enter the case study name')
            msg.setIcon(QMessageBox.Warning)
            x=msg.exec_()
            print('nocase')
        else:
            col_1=pd.Series(['12:00 AM','12:30 AM','1:00 AM','1:30 AM','2:00 AM','2:30 AM',\
                             '3:00 AM','3:30 AM','4:00 AM','4:30 AM','5:00 AM','5:30 AM',\
                            '6:00 AM','6:30 AM','7:00 AM','7:30 AM','8:00 AM','8:30 AM',\
                        '9:00 AM','9:30 AM','10:00 AM','10:30 AM','11:00 AM','11:30 AM',\
                        '12:00 PM','12:30 PM','1:00 PM','1:30 PM','2:00 PM','2:30 PM',\
                        '3:00 PM','3:30 PM','4:00 PM','4:30 PM','5:00 PM','5:30 PM',\
                        '6:00 PM','6:30 PM','7:00 PM','7:30 PM','8:00 PM','8:30 PM',\
                        '9:00 PM','9:30 PM','10:00 PM','10:30 PM','11:00 PM','11:30 PM'])
            app.ActivateProject(case_name)
    
            prj = app.GetActiveProject()
            
            
            filename = prj.GetAttribute("loc_name")
            loads = app.GetCalcRelevantObjects('*.ElmLod')
            load_dict = {}
            for i,load in enumerate(loads):
              # print(i, load.loc_name)
              load_dict[load.loc_name] = load

            #get all buses
            buses = app.GetCalcRelevantObjects('*.ElmTerm')
            bus_dict = {}
            for bus in buses:
              bus_dict[bus.loc_name] = bus

            #get all lines
            lines=app.GetCalcRelevantObjects('*.ElmLne')
            line_dict={}
            for line in lines:
              line_dict[line.loc_name] = line
              
            #get all static generator
            ders=app.GetCalcRelevantObjects('*.ElmGenstat')
            ders_dict = {}
            for i,der in enumerate(ders):
              ders_dict[der.loc_name] = der


            #retrieve load-flow object
            ldf = app.GetFromStudyCase("ComLdf")

            ldf.iopt_net = 0
             #execute load flow
            ldf.Execute()
            
            all_load_bus=[]
            all_load_values=[]
            all_pf=[]
            
            for n, load_key in enumerate(load_dict.keys()):
                load_value=load_dict[load_key].GetAttribute("m:P:bus1")
                all_load_bus.append(load_key)
                all_load_values.append(load_value)
                all_pf.append(load_dict[load_key].GetAttribute("m:cosphi:bus1"))
            all_pf=np.array(all_pf).reshape(-1,1)
            all_load_values=np.array(all_load_values)
            participation_factor=(all_load_values/sum(all_load_values)).reshape(-1,1)
            
            percentage_of_willingness_ess=float(self.percentage_of_willingness_ess.text())/100
            
            ess_capacity=float(self.total_ess_capacity.text())
            print(case_name)
            
            ESS_profile=np.ones(self.EV_profile.shape)*ess_capacity*percentage_of_willingness_ess
            main_container=[]
            for i in range(self.Load_profile.shape[0]):
                load_temp=np.kron(self.Load_profile[i,:],participation_factor)
                pv_temp=self.PV_profile[i,:]
                ev_temp=self.EV_profile[i,:]
                wind_temp=self.Wind_profile[i,:]
                ess_temp=ESS_profile[i,:]
                
                sub_container=[]
                for j in range(self.Load_profile.shape[1]):
                    
                    ders_dict['Chepstowe Wind Farm'].SetAttribute("pgini", wind_temp[j])
                    ders_dict['PV_Ballarat'].SetAttribute("pgini",  pv_temp[j])
                    ders_dict['EV_Ballarat'].SetAttribute("pgini",  ev_temp[j])
                    ders_dict['BESS_Ballarat'].SetAttribute("pgini",  ess_temp[j])
                    for n,load_key in enumerate(load_dict.keys()):
                        load_dict[load_key].SetAttribute("plini", load_temp[n,j] )
                        load_dict[load_key].SetAttribute("qlini", load_temp[n,j]*np.tan(np.arccos(all_pf[n][0])) )
                    #retrieve load-flow object
                    ldf = app.GetFromStudyCase("ComLdf")

                    ldf.iopt_net = 0
                     #execute load flow
                    result=ldf.Execute()
                    if result != 0:
                
                        print("Load flow did not converge at day "+str(i+1)+ " j="+str(j+1))
                        pass
                    else:
                        print("Load flow converged successfully at day "+str(i+1)+ " j="+str(j+1))
                    
                    overvoltage_bus_name=[]
                    over_voltage_value=[]

                    undervoltage_bus_name=[]
                    undervoltage_value=[]
                    violated_loading_line_name=[]
                    violated_loading_value=[]

                    for n,bus_key in enumerate(bus_dict.keys()):


                      bus_voltage=bus_dict[bus_key].GetAttribute("m:u1")
                    
                      if bus_voltage<0.94:
                          undervoltage_bus_name.append(bus_key)
                          undervoltage_value.append(bus_voltage)
                      if bus_voltage>1.05:
                          overvoltage_bus_name.append(bus_key)
                          over_voltage_value.append(bus_voltage)

                    for n, line_key in enumerate(line_dict.keys()):
                        line_loading=line_dict[line_key].GetAttribute("c:loading")
                        
                        if line_loading>100:
                            violated_loading_line_name.append(line_key)
                            violated_loading_value.append(line_loading)
                    undervoltage_bus_name=pd.Series(undervoltage_bus_name)
                    undervoltage_value=pd.Series(undervoltage_value)
                    overvoltage_bus_name=pd.Series(overvoltage_bus_name)
                    over_voltage_value=pd.Series(over_voltage_value)
                    violated_loading_line_name=pd.Series(violated_loading_line_name)
                    violated_loading_value=pd.Series(violated_loading_value)
                    
                    df_temp=pd.concat([undervoltage_bus_name,undervoltage_value,overvoltage_bus_name,over_voltage_value, violated_loading_line_name, violated_loading_value],axis=1)
                    df_temp=df_temp.set_axis(['Undervoltage bus','Value','Overvoltage bus','Value','Overloaded lines','Value (in %)'], axis=1)
                    sub_container.append(df_temp)
                main_container.append(sub_container)

            for i in range(len(main_container)):
                with pd.ExcelWriter("Security check day "+str(i+1)+".xlsx") as writer:
                    
                    for j in range(len(main_container[i])):
                        
                        main_container[i][j].to_excel(writer,sheet_name=col_1[j].replace(":","-"), index=False)
            msg = QMessageBox()
            msg.setWindowTitle('Info')
            msg.setText('Security check completed. Please see the excel files.')
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            
            
    def show_contact_info(self):
        msg = QMessageBox()
        msg.setWindowTitle('Contact information')
        msg.setText('If you have any questions, please reach out to Tan Pham (pntan.iac@gmail.com) or BM Amin (b.amin@federation.edu.au)')
        msg.setIcon(QMessageBox.Information)
        x=msg.exec_()
        
    def plot_stage_1(self):
        if self.flag_file_ready==0:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText('Cannot access the Excel file. Please select your Excel file')
            msg.setIcon(QMessageBox.Critical)
            x=msg.exec_()
        else:
            
            df=pd.read_excel(self.file_address)
            net_load=df[df.columns[0]]
            PV=df[df.columns[1]]
            Wind=df[df.columns[2]]
            EV=df[df.columns[3]]
    
            plt.figure(1)
            plt.plot(net_load,'r')
            plt.title('Net load')
            plt.xlabel ('Half hourly time instances')
            plt.ylabel ('MW')
            plt.show()
    
            plt.figure(2)
            plt.plot(PV,'g')
            plt.title('PV')
            plt.xlabel ('Half hourly time instances')
            plt.ylabel ('MW')
            plt.show()
    
            plt.figure(3)
            plt.plot(Wind,'b')
            plt.title('Wind')
            plt.xlabel ('Half hourly time instances')
            plt.ylabel ('MW')
            plt.show()
    
            plt.figure(4)
            plt.plot(EV,'k')
            plt.title('EV')
            plt.xlabel ('Half hourly time instances')
            plt.ylabel ('MW')
            plt.show()
            # df.plot
        
    
    def gen_rep_profile(self):
        df=pd.read_excel(self.file_address)
        # Filter the DataFrame to include only negative values
        negative_values = df[df < 0]
        
        # Convert negative values to their absolute values using NumPy
        abs_negative_values = np.abs(negative_values)
        
        # Find the maximum value among the absolute values
        max_abs_negative_value = abs_negative_values.max().max()
        
        print("Maximum absolute value of negative values:", max_abs_negative_value)
        
        bias = max_abs_negative_value


        # Add bias to the first column
        df.iloc[:, 0] += bias
        
        
        df_numpy=df.T.to_numpy()
        
        
        
        ## 5671 - 5677 are nan, replaced by zeros
        
        participation_factor_matrix=np.divide(df_numpy,df_numpy.sum(axis=0))
        
        # Filter out nan
        
        participation_factor_matrix=np.nan_to_num(participation_factor_matrix,0)
        
        # participation_factor_matrix_average=participation_factor_matrix.T.mean(axis=0)
        
        step=48
        
        participation_factor_matrix_average=np.zeros((df.shape[1],step))
        
        temp_sum=np.zeros(4)
        count=0
        for k in range(step):
            for i in range(int(df.shape[0]/step)):
                # print(count+step*i)
                temp_sum+=participation_factor_matrix[:,count+step*i]
            participation_factor_matrix_average[:,k]=temp_sum/(df.shape[0]/step)
            temp_sum=np.zeros(4)
            count+=1
        
            
 
        
        
        
        
        ## Time aggregation
        
        Average_centroid_of_time_series=np.nan_to_num(df_numpy.sum(axis=0),0)
        
        

        
        Stacked_matrix= np.zeros((int(df.shape[0]/step),step))
        
        count=1
        for i in range(int(df.shape[0]/step)):
            Stacked_matrix[i,:]=Average_centroid_of_time_series[step*i:step*(i+1)]
            
        # Sihoulet criteria calculation
        sillhoute_scores = []
        n_cluster_list = np.arange(2,31).astype(int)

        X = Stacked_matrix.copy()
            
        # Very important to scale!
        sc = MinMaxScaler()
        X = sc.fit_transform(X)

        for n_cluster in n_cluster_list:
            
            kmeans = KMeans(n_clusters=n_cluster)
            cluster_found = kmeans.fit_predict(X)
            sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
            
        
        plt.figure(figsize=(13,8),dpi=80)
        plt.plot(n_cluster_list,sillhoute_scores,'b')
        
        font = {'family': 'Calibri',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
        plt.xlabel('Number of clusters', fontdict=font)
        plt.ylabel('Average sillhoute', fontdict=font)
        plt.xticks([i for i in range(0, len(n_cluster_list)+2)])
        
        Optimal_NumberOf_Components=n_cluster_list[sillhoute_scores.index(max(sillhoute_scores))]
        # print(n_cluster_list)
        # print ("Optimal number of components is:")
        # print (Optimal_NumberOf_Components)
        
        self.dist_num_cluster.display(Optimal_NumberOf_Components)
        self.opt_num_cluster=Optimal_NumberOf_Components
        
        return self.opt_num_cluster
        
        
        
        
        
       
        
    def gen_rep_profile_manual(self):
        
       df=pd.read_excel(self.file_address)
       # Filter the DataFrame to include only negative values
       negative_values = df[df < 0]
       
       # Convert negative values to their absolute values using NumPy
       abs_negative_values = np.abs(negative_values)
       
       # Find the maximum value among the absolute values
       max_abs_negative_value = abs_negative_values.max().max()
       
       
       
       bias = max_abs_negative_value


       # Add bias to the first column
       df.iloc[:, 0] += bias
       
       
       df_numpy=df.T.to_numpy()
       
       
       
       ## 5671 - 5677 are nan, replaced by zeros
       
       participation_factor_matrix=np.divide(df_numpy,df_numpy.sum(axis=0))
       
       # Filter out nan
       
       participation_factor_matrix=np.nan_to_num(participation_factor_matrix,0)
       
       # participation_factor_matrix_average=participation_factor_matrix.T.mean(axis=0)
       
       step=48
       
       participation_factor_matrix_average=np.zeros((df.shape[1],step))
       
       temp_sum=np.zeros(4)
       count=0
       for k in range(step):
           for i in range(int(df.shape[0]/step)):
               # print(count+step*i)
               temp_sum+=participation_factor_matrix[:,count+step*i]
           participation_factor_matrix_average[:,k]=temp_sum/(df.shape[0]/step)
           temp_sum=np.zeros(4)
           count+=1
       
           
      
       
       
       
       
       ## Time aggregation
       
       Average_centroid_of_time_series=np.nan_to_num(df_numpy.sum(axis=0),0)
       Stacked_matrix= np.zeros((int(df.shape[0]/step),step))
       count=1
       for i in range(int(df.shape[0]/step)):
           Stacked_matrix[i,:]=Average_centroid_of_time_series[step*i:step*(i+1)]   
        
       num_cluster=self.opt_num_cluster
       # print(int(num_cluster))
       kmeans = KMeans(n_clusters=int(num_cluster)).fit(Stacked_matrix)
       
       Centroids=kmeans.cluster_centers_

       
       ## Reverse back
       Load_profile=np.multiply(Centroids, participation_factor_matrix_average[0,:])
       
       Load_profile = Load_profile - bias
       PV_profile=np.multiply(Centroids,participation_factor_matrix_average[1,:])
       Wind_profile=np.multiply(Centroids,participation_factor_matrix_average[2,:])
       EV_profile=np.multiply(Centroids,participation_factor_matrix_average[3,:])
       
       
       self.Load_profile = Load_profile
       self.PV_profile=  PV_profile
       self.Wind_profile= Wind_profile
       self.EV_profile= EV_profile
       
       plt.figure(figsize=(13,8),dpi=80)
       for i in range(Load_profile.shape[0]):
           plt.plot(Load_profile[i,:])
       plt.xlabel('Half hourly time series')
       plt.ylabel('Power, MW')
       plt.title('Representative Net Load')
       legend_holder=[]
       for i in range(Load_profile.shape[0]):
           legend_holder.append('Day '+str(i+1))
       plt.legend(legend_holder)
       plt.xticks([i for i in range(0, Load_profile.shape[1]+2)])
       plt.show()
       
       
       plt.figure(figsize=(13,8),dpi=80)
       for i in range(PV_profile.shape[0]):
           plt.plot(PV_profile[i,:])
       plt.xlabel('Half hourly time series')
       plt.ylabel('Power, MW')
       plt.title('Representative PV profile')
       legend_holder=[]
       for i in range(Load_profile.shape[0]):
           legend_holder.append('Day '+str(i+1))
       plt.legend(legend_holder)
       plt.xticks([i for i in range(0, Load_profile.shape[1]+2)])
       plt.show()
       
       
       plt.figure(figsize=(13,8),dpi=80)
       for i in range(Wind_profile.shape[0]):
           plt.plot(Wind_profile[i,:])
       plt.xlabel('Half hourly time series')
       plt.ylabel('Power, MW')
       plt.title('Representative Wind profile')
       legend_holder=[]
       for i in range(Load_profile.shape[0]):
           legend_holder.append('Day '+str(i+1))
       plt.legend(legend_holder)
       plt.xticks([i for i in range(0, Load_profile.shape[1]+2)])
       plt.show()
       
       plt.figure(figsize=(13,8),dpi=80)
       for i in range(EV_profile.shape[0]):
           plt.plot(EV_profile[i,:])
       plt.xlabel('Half hourly time series')
       plt.ylabel('Power, MW')
       plt.title('Representative EV profile')
       legend_holder=[]
       for i in range(Load_profile.shape[0]):
           legend_holder.append('Day '+str(i+1))
       plt.legend(legend_holder)
       plt.xticks([i for i in range(0, Load_profile.shape[1]+2)])
       plt.show() 
         
       self.flag_save=1
       return self.Load_profile,self.Wind_profile,self.PV_profile,self.EV_profile,self.flag_save
           
    def save_rep_profile(self):
        
        if self.flag_save ==0:
            msg = QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Please generate the profiles first!')
            msg.setIcon(QMessageBox.Warning)
            x=msg.exec_()
        else:
            data_frame1=pd.DataFrame(self.Load_profile)
            data_frame2=pd.DataFrame(self.Wind_profile)
            data_frame3=pd.DataFrame(self.PV_profile)
            data_frame4=pd.DataFrame(self.EV_profile)
            
            with pd.ExcelWriter("Representative_profiles.xlsx") as writer:
       
                # use to_excel function and specify the sheet_name and index 
                # to store the dataframe in specified sheet
                data_frame1.to_excel(writer, sheet_name="Load", index=False)
                data_frame2.to_excel(writer, sheet_name="Wind", index=False)
                data_frame3.to_excel(writer, sheet_name="PV", index=False)
                data_frame4.to_excel(writer, sheet_name="EV", index=False)
         
             
            msg = QMessageBox()
            msg.setWindowTitle('Information')
            msg.setText('File saved!')
            msg.setIcon(QMessageBox.Information)
            x=msg.exec_()
        
    def ancillary_calculation(self):
        percentage_of_customer_dr=float(self.percentage_of_customer_dr.text())/100
        
        percentage_of_willingness=float(self.percentage_of_willingness.text())/100
        percentage_of_willingness_pv=float(self.percentage_of_willingness_pv.text())/100
        percentage_of_willingness_wind=float(self.percentage_of_willingness_wind.text())/100
        percentage_of_willingness_ev=float(self.percentage_of_willingness_ev.text())/100
        percentage_of_willingness_ess=float(self.percentage_of_willingness_ess.text())/100
        
        ess_capacity=float(self.total_ess_capacity.text())
        
        DR_up_reserve=float(self.dr_up_reserve.text())/100
        DR_down_reserve=float(self.dr_down_reserve.text())/100
        
        PV_up_reserve=float(self.pv_up_reserve.text())/100
        PV_down_reserve=float(self.pv_down_reserve.text())/100
        
        Wind_up_reserve=float(self.wind_up_reserve.text())/100
        Wind_down_reserve=float(self.wind_down_reserve.text())/100
        
        EV_up_reserve=float(self.ev_up_reserve.text())/100
        EV_down_reserve=float(self.ev_down_reserve.text())/100
        
        ESS_up_reserve=float(self.ess_up_reserve.text())/100
        ESS_down_reserve=float(self.ess_down_reserve.text())/100
        
        sys.path.append(self.python_folder_address)
        import powerfactory as pf
        import os
        # os.environ["PATH"] = r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP3;"+ os.environ["PATH"]
        app = pf.GetApplication()
        app.ClearOutputWindow()
        case_name=self.stage_2_case_name.text()
        
        app.ActivateProject(case_name)

        prj = app.GetActiveProject()

        filename = prj.GetAttribute("loc_name")
        loads = app.GetCalcRelevantObjects('*.ElmLod')
        load_dict = {}
        for i,load in enumerate(loads):
          # print(i, load.loc_name)
          load_dict[load.loc_name] = load
          
        #get all buses
        buses = app.GetCalcRelevantObjects('*.ElmTerm')
        bus_dict = {}
        for bus in buses:
          bus_dict[bus.loc_name] = bus

        #get all lines
        lines=app.GetCalcRelevantObjects('*.ElmLne')
        line_dict={}
        for line in lines:
          line_dict[line.loc_name] = line

        #retrieve load-flow object
        ldf = app.GetFromStudyCase("ComLdf")

        ldf.iopt_net = 0
         #execute load flow
        ldf.Execute()
        
        all_voltage=[]
        all_busname=[]
        all_linename=[]
        all_line_loading=[]

        all_load_bus=[]
        all_load_values=[]
        all_pf=[]

      

        for n,bus_key in enumerate(bus_dict.keys()):
          
          
          bus_voltage=bus_dict[bus_key].GetAttribute("m:u1")
          all_voltage.append(bus_voltage)
          all_busname.append(bus_key)
          
          
        for n, line_key in enumerate(line_dict.keys()):
            line_loading=line_dict[line_key].GetAttribute("c:loading")
            all_linename.append(line_key)
            all_line_loading.append(line_loading)
            

        for n, load_key in enumerate(load_dict.keys()):
            load_value=load_dict[load_key].GetAttribute("m:P:bus1")
            all_load_bus.append(load_key)
            all_load_values.append(load_value)
            all_pf.append(load_dict[load_key].GetAttribute("m:cosphi:bus1"))
            
        all_load_values=np.array(all_load_values)
        participation_factor=(all_load_values/sum(all_load_values)).reshape(-1,1)
            

        Load_profile, PV_profile, Wind_profile, EV_profile=self.Load_profile, self.PV_profile, self.Wind_profile, self.EV_profile


        tensor_load_P=np.zeros((Load_profile.shape[0],participation_factor.shape[0],Load_profile.shape[1]))
        tensor_load_Q=tensor_load_P.copy()
        all_pf=np.array(all_pf).reshape(-1,1)

        dr_array_P=np.zeros(Load_profile.shape)
        dr_array_Q=np.zeros(Load_profile.shape)

        for i in range(tensor_load_P.shape[0]):
            tensor_load_P[i]=participation_factor*Load_profile[i]
            tensor_load_Q[i]=tensor_load_P[i]*np.tan(np.arccos(all_pf))


        

        for i in range(tensor_load_P.shape[0]):
            for j in range(tensor_load_P.shape[2]):
            
                test=tensor_load_P[i,:,j]
                test_Q=tensor_load_Q[i,:,j]
                
                test_sort=np.sort(test)[::-1]
                test_sort_Q=np.sort(test_Q)[::-1]
                
                test_sort_up=test_sort[0:len(test_sort)//2]
                test_sort_down=test_sort[len(test_sort)//2:len(test_sort)]
                
                test_sort_up_Q=test_sort_Q[0:len(test_sort_Q)//2]
                test_sort_down_Q=test_sort_Q[len(test_sort_Q)//2:len(test_sort_Q)]
                
                
                sampled_up=np.random.choice(test_sort_up, int(np.ceil(len(test_sort_up)*percentage_of_customer_dr)),replace=False)
                sampled_down=np.random.choice(test_sort_down, int(np.ceil(len(test_sort_down)*percentage_of_customer_dr)),replace=False)
                
                sampled_up_Q=np.random.choice(test_sort_up_Q, int(np.ceil(len(test_sort_up_Q)*percentage_of_customer_dr)),replace=False)
                sampled_down_Q=np.random.choice(test_sort_down_Q, int(np.ceil(len(test_sort_down_Q)*percentage_of_customer_dr)),replace=False)
                
                dr_array_P[i][j]=sum((sampled_up+sampled_down))
                dr_array_Q[i][j]=sum((sampled_up_Q+sampled_down_Q))

        dr_array_P=dr_array_P.T
        dr_array_Q=dr_array_Q.T

        

        avai_dr_frequency_supp=dr_array_P*percentage_of_willingness
        avai_dr_for_up_reserve=avai_dr_frequency_supp*DR_up_reserve
        avai_dr_for_down_reserve=avai_dr_frequency_supp*DR_down_reserve
        avai_dr_voltage_supp=dr_array_Q*percentage_of_willingness

        dr_array_P=pd.DataFrame(dr_array_P)
        dr_array_Q=pd.DataFrame(dr_array_Q)
        avai_dr_frequency_supp=pd.DataFrame(avai_dr_frequency_supp)
        avai_dr_for_up_reserve=pd.DataFrame(avai_dr_for_up_reserve)
        avai_dr_for_down_reserve=pd.DataFrame(avai_dr_for_down_reserve)
        avai_dr_voltage_supp=pd.DataFrame(avai_dr_voltage_supp)


        col_1=pd.Series(['12:00 AM','12:30 AM','1:00 AM','1:30 AM','2:00 AM','2:30 AM',\
                         '3:00 AM','3:30 AM','4:00 AM','4:30 AM','5:00 AM','5:30 AM',\
                        '6:00 AM','6:30 AM','7:00 AM','7:30 AM','8:00 AM','8:30 AM',\
                    '9:00 AM','9:30 AM','10:00 AM','10:30 AM','11:00 AM','11:30 AM',\
                    '12:00 PM','12:30 PM','1:00 PM','1:30 PM','2:00 PM','2:30 PM',\
                    '3:00 PM','3:30 PM','4:00 PM','4:30 PM','5:00 PM','5:30 PM',\
                    '6:00 PM','6:30 PM','7:00 PM','7:30 PM','8:00 PM','8:30 PM',\
                    '9:00 PM','9:30 PM','10:00 PM','10:30 PM','11:00 PM','11:30 PM'])

        with pd.ExcelWriter("Results_case_1.xlsx") as writer:
            for i in range(dr_array_P.shape[1]):
                
                df_test=pd.concat([col_1,dr_array_P[i],avai_dr_frequency_supp[i],\
                               avai_dr_for_up_reserve[i],avai_dr_for_down_reserve[i],\
                               avai_dr_voltage_supp[i]],axis=1)
            
                df_test=df_test.set_axis(['Time','Total interested DR loads','Available DR for frequency support',	'Up reserve',	'Down reserve',	'Available DR for voltage support'
                ], axis=1)
                
                df_test.to_excel(writer, sheet_name="Rep. Day" + str(i+1), index=False)
        
     

        total_anci_from_pv=pd.DataFrame(PV_profile.T*percentage_of_willingness)
        up_reseve_from_pv=pd.DataFrame(total_anci_from_pv*PV_up_reserve)
        down_reseve_from_pv=pd.DataFrame(total_anci_from_pv*PV_down_reserve)

       


        
        total_anci_from_wind=pd.DataFrame(Wind_profile.T*percentage_of_willingness)
        up_reseve_from_wind=pd.DataFrame(total_anci_from_wind*Wind_up_reserve)
        down_reseve_from_wind=pd.DataFrame(total_anci_from_wind*Wind_down_reserve)


        avai_voltage_supp_PV=total_anci_from_pv*np.tan(np.arccos(0.8))
        avai_voltage_supp_Wind=total_anci_from_wind*np.tan(np.arccos(0.8))

        # Total anci service for case 2
        total_anci_service=avai_dr_frequency_supp+total_anci_from_pv+total_anci_from_wind

        total_up_reserve=avai_dr_for_up_reserve+up_reseve_from_pv+up_reseve_from_wind
        total_down_reserve=avai_dr_for_down_reserve+down_reseve_from_pv+down_reseve_from_wind

        total_voltage_supp_case2=avai_dr_voltage_supp+avai_voltage_supp_PV+avai_voltage_supp_Wind

        with pd.ExcelWriter("Results_case_2.xlsx") as writer:
            for i in range(dr_array_P.shape[1]):

                df_test=pd.concat([col_1,dr_array_P[i],avai_dr_frequency_supp[i],avai_dr_for_up_reserve[i],avai_dr_for_down_reserve[i],\
                               
                                   total_anci_from_pv[i],up_reseve_from_pv[i], down_reseve_from_pv[i],\
                                total_anci_from_wind[i],up_reseve_from_wind[i],down_reseve_from_wind[i],\
                            total_anci_service[i], total_up_reserve[i],total_down_reserve[i],\
                                
                                avai_dr_voltage_supp[i],avai_voltage_supp_PV[i],avai_voltage_supp_Wind[i],total_voltage_supp_case2[i]  ],axis=1)

                df_test=df_test.set_axis(['Time','Interested DR loads','Available DR for frequency support','Up reserve','Down reserve','Ancillary services from PV','Up reserve from PV','Down Reserve from PV','Ancillary services from WIND','Up reserve from WIND','Down Reserve from WIND','Total ancillary service','Total up reserve','Total down reserve','Available voltage support from DR','Available voltage support from PV','Available voltage support from WIND','Total available voltage support'
                ], axis=1)

                df_test.to_excel(writer, sheet_name="Rep. Day" + str(i+1), index=False)
        
        anci_from_ev=pd.DataFrame(EV_profile.T*percentage_of_willingness_ev)
        up_reserve_from_ev=pd.DataFrame(anci_from_ev*EV_up_reserve)
        down_reserve_from_ev=pd.DataFrame(anci_from_ev*EV_down_reserve) 


      

        anci_from_ess=pd.DataFrame(np.ones(EV_profile.T.shape)*ess_capacity*percentage_of_willingness_ess)

        


        up_reserve_from_ess=pd.DataFrame(anci_from_ess*ESS_up_reserve)
        down_reserve_from_ess=pd.DataFrame(anci_from_ess*ESS_down_reserve)

        total_anci_service_new=total_anci_service+anci_from_ev+anci_from_ess
        total_up_reserve_new=total_up_reserve+up_reserve_from_ev+up_reserve_from_ess
        total_down_reserve_new=total_down_reserve+down_reserve_from_ev+down_reserve_from_ess


        avai_voltage_supp_ev=anci_from_ev*np.tan(np.arccos(0.8))
        avai_voltage_supp_ess=anci_from_ess*np.tan(np.arccos(0.8))
        total_voltage_supp_case3=total_voltage_supp_case2+avai_voltage_supp_ev+avai_voltage_supp_ess

        with pd.ExcelWriter("Results_case_3.xlsx") as writer:
            for i in range(dr_array_P.shape[1]):
                
                df_test=pd.concat([col_1,dr_array_P[i],avai_dr_frequency_supp[i],avai_dr_for_up_reserve[i],avai_dr_for_down_reserve[i],\
                                   
                                   total_anci_from_pv[i],up_reseve_from_pv[i], down_reseve_from_pv[i],\
                                  total_anci_from_wind[i],up_reseve_from_wind[i],down_reseve_from_wind[i],\
                                   
                                   anci_from_ev[i],up_reserve_from_ev[i],down_reserve_from_ev[i],\
                                   anci_from_ess[i],up_reserve_from_ess[i],down_reserve_from_ess[i],\
                                       
                                   total_anci_service_new[i], total_up_reserve_new[i], total_down_reserve_new[i],\
                                       
                                   avai_dr_voltage_supp[i],avai_voltage_supp_PV[i],avai_voltage_supp_Wind[i], avai_voltage_supp_ev[i], avai_voltage_supp_ess[i], total_voltage_supp_case3[i]
                                       
                                   ],axis=1)

              

                df_test=df_test.set_axis(['Time','Interested DR loads','Available DR for frequency support',	'Up reserve','Down reserve','Ancillary services from PV','Up reserve from PV','Down Reserve from PV','Ancillary services from WIND','Up reserve from WIND','Down Reserve from WIND','Ancillary service from EV','Up reserve from EV','Down reserve from EV','Ancillary service from ESS','Up reserve from ESS','Down reserve from ESS','Total ancillary service','Total up reserve','Total down reserve','Available voltage support from DR','Available voltage support from PV','Available voltage support from WIND','Available voltage support from EV','Available voltage support from ESS','Total available voltage support'

                ], axis=1)

                df_test.to_excel(writer, sheet_name="Rep. Day" + str(i+1), index=False)
        
        msg = QMessageBox()
        msg.setWindowTitle('Information')
        msg.setText('Calculation completed, file saved!')
        msg.setIcon(QMessageBox.Information)
        x=msg.exec_()
      
if __name__ == "__main__":
    # Handle high resolution displays:
    
    app=QApplication(sys.argv)
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    mainwindow=Solver()
    screen_geometry = QDesktopWidget().screenGeometry()
    
      # Set the window size to fit the screen
      # self.setGeometry(0, 0, screen_geometry.width(), screen_geometry.height())
    widget=QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(screen_geometry.width())
    widget.setFixedHeight(screen_geometry.height())
    widget.setWindowTitle('Ancillary services calculator')
    widget.show()
    app.exec_()

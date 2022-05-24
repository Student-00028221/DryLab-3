import numpy as np
import math
import PySimpleGUI as sg
import pandas as pd
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from spatialmath import SE3
import matplotlib.pyplot as plt

# GUI code

sg.theme('DarkBlack')

# Excel Read Code

EXCEL_FILE = 'Spherical Manipulator RRP Calculator Design Data.xlsx'
df = pd.read_excel(EXCEL_FILE)

# Lay-out code

Main_layout = [
    [sg.Push(), sg.Text('Spherical RRP MEXE CALCULATOR', font = ("Lucida Sans Unicode", 15)), sg.Push()],
    [sg.Push(), sg.Button('Start the Calculation', font = ("Lucida Sans Unicode", 15), size=(36,0), button_color=('white', 'gray')), sg.Push()],
    [sg.Text('Forward Kinematics Calculator', font = ("Lucida Sans", 12))],
    [sg.Text('Fill out the following fields:', font = ("Lucida Sans", 10))],
    
    [sg.Text('a1 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a1', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
    sg.Text('T1 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='T1', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
    sg.Push(),sg.Button('Jacobian Matrix (J)', bind_return_key=True,disabled=True, font = ("Lucida Sans Unicode", 12), size=(15,0), button_color=('black', 'lightgray')),
    sg.Button('Det(J)', bind_return_key=True,disabled=True, font = ("Lucida Sans Unicode", 12), size=(15,0), button_color=('black', 'lightgray')), sg.Push()],

    [sg.Text('a2 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a2', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
    sg.Text('T2 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='T2', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
    sg.Push(),sg.Button('Inverse of J', bind_return_key=True,disabled=True, font = ("Lucida Sans Unicode", 12), size=(15,0), button_color=('black', 'lightgray')),
    sg.Button('Transpose of J', bind_return_key=True,disabled=True, font = ("Lucida Sans Unicode", 12), size=(15,0), button_color=('black', 'lightgray')), sg.Push()],

    [sg.Text('a3 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a3', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
    sg.Text('d3 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='d3', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10))],

    [sg.Button('Solve Forward Kinematics', bind_return_key=True,disabled=True, tooltip='Go to "Start the Calculation"!', font = ("Lucida Sans Unicode", 12), button_color=('black', 'white')),
    sg.Button('Inverse Kinematics', bind_return_key=True,disabled=True, font = ("Lucida Sans Unicode", 12), size=(23,0), button_color=('lightgray', 'gray')), sg.Push()],
    
    [sg.Frame('Position Vector: ',[[
        sg.Text('X =', font = ("Lucida Sans", 10)),sg.InputText('0', key='X', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10)),
        sg.Text('Y =', font = ("Lucida Sans", 10)),sg.InputText('0', key='Y', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10)),
        sg.Text('Z =', font = ("Lucida Sans", 10)),sg.InputText('0', key='Z', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10))]])],

    [sg.Frame('H0_3 Transformation Matrix = ', [[sg.Output(size=(61,15), key = '_output_')]]),
    sg.Push(), sg.Image('SphericalM2.gif', key='_IMAGE_'), sg.Push()],
    [sg.Submit(font = ("Lucida Sans", 10)), sg.Button('Reset', font = ("Lucida Sans", 10)), sg.Exit(font = ("Lucida Sans", 10))]]

window = sg.Window('Spherical-RRP Manipulator Forward Kinematics', Main_layout, resizable=True)

# Inverse Kinematics Inverse Function

def inverse_kinematics_window():
     # GUI code

    sg.theme('DarkBlack')

     # Excel Read Code

    EXCEL_FILE = 'INVERSE KINEMATICS DATA.xlsx'
    ik_df = pd.read_excel(EXCEL_FILE)

    IK_layout = [
        [sg.Push(), sg.Text('Spherical RRP Inverse Kinematics', font = ("Lucida Sans Unicode", 15)), sg.Push()],
        [sg.Text('Fill out the following fields:', font = ("Lucida Sans", 10))],
        
        [sg.Text('a1 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a1', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
        sg.Text('X =', font = ("Lucida Sans", 10)),sg.InputText('0', key='X', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10))],
        [sg.Text('a2 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a2', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
        sg.Text('Y =', font = ("Lucida Sans", 10)),sg.InputText('0', key='Y', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10))],

        [sg.Text('a3 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='a3', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10)),
        sg.Text('Z =', font = ("Lucida Sans", 10)),sg.InputText('0', key='Z', size=(20,10)), sg.Text('mm', font = ("Lucida Sans", 10))],
        
        [sg.Button('Solve Inverse Kinematics',  tooltip='Go to "Start the Calculation"!', font = ("Lucida Sans Unicode", 12), button_color=('black', 'white'))],
            
        [sg.Frame('Joint Variables Value: ',[[
            sg.Text('T1 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='ik_T1', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10)),
            sg.Text('T2 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='ik_T2', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10)),
            sg.Text('d3 =', font = ("Lucida Sans", 10)),sg.InputText('0', key='ik_d3', size=(10,1)), sg.Text('mm', font = ("Lucida Sans", 10))]])],

        [sg.Submit(font = ("Lucida Sans", 10)), sg.Button('Reset', font = ("Lucida Sans", 10)), sg.Exit(font = ("Lucida Sans", 10))]
        ]

    inverse_kinematics_window = sg.Window('Spherical-RRP Manipulator Forward Kinematics', IK_layout)

    while True:
        event,values = inverse_kinematics_window.read(0)
            
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == 'Solve Inverse Kinematics':
                
            # Inverse Kinematic Codes
            
            # link lengths in cm
            a1 = float(values['a1'])
            a2 = float(values['a2'])
            a3 = float(values['a3'])

            # PARAMETERS
            Sphe_Modern = DHRobot([
                RevoluteDH(a1,0,(90/180)*np.pi,0,qlim=[(-180/180)*np.pi,(180/180)*np.pi]),
                RevoluteDH(0,a2,(90/180)*np.pi,(90/180)*np.pi,qlim=[(-180/180)*np.pi,(180/180)*np.pi]),
                PrismaticDH(0,0,0,a3,qlim=[0,100]),
                ], name='Spherical')

            # Joint Variable (Thetas in degrees & dinstance in cm)
            X = float(values['X'])
            Y = float(values['Y'])
            Z = float(values['Z'])

            T = SE3(X , Y, Z)

            Ikine = Sphe_Modern.ikine_LM(T)
            Ik_arr = Ikine[0]

            try:
                theta1 = (((np.arctan(Y/X))*180.0)/np.pi)
            except:
                theta1 = -1 #NAN
                sg.popup('Warning! Present values causes error.')
                sg.popup('Restart the GUI then assign proper values')
                break

            theta1 = Ik_arr[0]
            theta2 = Ik_arr[1]
            d3 = Ik_arr[2]

            Th1 = (theta1)*(180.0/np.pi) # Theta 1 from radians
            Th2 = (theta2)*(180.0/np.pi) # Theta 2 from radians

            inverse_kinematics_window['ik_T1'].update(np.around(Th1,5))
            inverse_kinematics_window['ik_T2'].update(np.around(Th2,5))
            inverse_kinematics_window['ik_d3'].update(np.around(d3))

        if event == 'Reset' :

            inverse_kinematics_window['a1'].update(0)
            inverse_kinematics_window['a2'].update(0)
            inverse_kinematics_window['a3'].update(0)
            inverse_kinematics_window['ik_T1'].update(0)
            inverse_kinematics_window['ik_T2'].update(0)
            inverse_kinematics_window['ik_d3'].update(0)
            inverse_kinematics_window['X'].update(0)
            inverse_kinematics_window['Y'].update(0)
            inverse_kinematics_window['Z'].update(0)

        if event == 'Submit' :
            ik_df = ik_df.append(values, ignore_index=True)
            ik_df.to_excel(EXCEL_FILE, index=False)
            sg.popup('Data Saved!')

    inverse_kinematics_window.close()

# Variable Codes for disabling buttons

disable_FK = window['Solve Forward Kinematics']
disable_J = window['Jacobian Matrix (J)']
disable_D = window['Det(J)']
disable_IV = window['Inverse of J']
disable_TJ = window['Transpose of J']
disable_IK = window['Inverse Kinematics']

while True:
    event,values = window.read(200)
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    window.Element('_IMAGE_').UpdateAnimation('SphericalM2.gif',  time_between_frames=50)

    if event == 'Reset' :

        window['a1'].update(0)
        window['a2'].update(0)
        window['a3'].update(0)
        window['T1'].update(0)
        window['T2'].update(0)
        window['d3'].update(0)
        window['X'].update(0)
        window['Y'].update(0)
        window['Z'].update(0)

        disable_FK.update(disabled=True)
        disable_J.update(disabled=True)
        disable_D.update(disabled=True)
        disable_IV.update(disabled=True)
        disable_TJ.update(disabled=True)
        disable_IK.update(disabled=True)

        window['_output_'].update('')

    if event == 'Start the Calculation' :
        disable_FK.update(disabled=False)
        disable_IK.update(disabled=False)

    if event == 'Solve Forward Kinematics' :
        
        # Forward Kinematic Codes
      
        # link lengths in cm
        a1 = float(values['a1'])
        a2 = float(values['a2'])
        a3 = float(values['a3'])

        # Joint Variable (Thetas in degrees & dinstance in cm)
        T1 = float(values['T1'])
        T2 = float(values['T2'])
        d3 = float(values['d3'])

        T1 = (T1/180.0)*np.pi  # Theta 1 in radian
        T2 = (T2/180.0)*np.pi  # Theta 2 in radian

        DHPT = [
            [T1,(90.0/180.0)*np.pi, 0, a1],
            [T2+(90.0/180.0)*np.pi, (90.0/180.0)*np.pi, a2, 0],
            [0, 0, 0, a3+d3],
            ]

        # D-H Notation Formula for HTM
        i = 0
        H0_1 = [
            [np.cos(DHPT[i][0]), -np.sin(DHPT[i][0])*np.cos(DHPT[i][1]), np.sin(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.cos(DHPT[i][0])],
            [np.sin(DHPT[i][0]), np.cos(DHPT[i][0])*np.cos(DHPT[i][1]), -np.cos(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.sin(DHPT[i][0])],
            [0, np.sin(DHPT[i][1]), np.cos(DHPT[i][1]), DHPT[i][3]],
            [0, 0, 0, 1],
            ]

        i = 1
        H1_2 = [
            [np.cos(DHPT[i][0]), -np.sin(DHPT[i][0])*np.cos(DHPT[i][1]), np.sin(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.cos(DHPT[i][0])],
            [np.sin(DHPT[i][0]), np.cos(DHPT[i][0])*np.cos(DHPT[i][1]), -np.cos(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.sin(DHPT[i][0])],
            [0, np.sin(DHPT[i][1]), np.cos(DHPT[i][1]), DHPT[i][3]],
            [0, 0, 0, 1],
            ]

        i = 2
        H2_3 = [
            [np.cos(DHPT[i][0]), -np.sin(DHPT[i][0])*np.cos(DHPT[i][1]), np.sin(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.cos(DHPT[i][0])],
            [np.sin(DHPT[i][0]), np.cos(DHPT[i][0])*np.cos(DHPT[i][1]), -np.cos(DHPT[i][0])*np.sin(DHPT[i][1]), DHPT[i][2]*np.sin(DHPT[i][0])],
            [0, np.sin(DHPT[i][1]), np.cos(DHPT[i][1]), DHPT[i][3]],
            [0, 0, 0, 1],
            ]

        # Transformation Matrices from base to end-effector
        #print("HO_1 = ")
        #print(np.matrix(H0_1))
        #print("H1_2 = ")
        #print(np.matrix(H1_2))
        #print("H2_3 = ")
        #print(np.matrix(H2_3))

        # Dot Product of H0_3 = HO_1*H1_2*H2_3
        H0_2 = np.dot(H0_1,H1_2)
        H0_3 = np.dot(H0_2,H2_3)

        # Transformation Matrix of the Manipulator
        print("H0_3 = ")
        print(np.matrix(H0_3))

        # Position Vector X Y Z

        X0_3 = H0_3[0,3]
        print("X = ", X0_3)

        Y0_3 = H0_3[1,3]
        print("Y = ", Y0_3)

        Z0_3 = H0_3[2,3]
        print("Z = ", Z0_3)
        
        # Disabler program 
        disable_J.update(disabled=False)
        disable_D.update(disabled=True)
        disable_IV.update(disabled=True)
        disable_TJ.update(disabled=True)

        # XYZ OUTPUT TO INPUT UPDATER
        window['X'].update(X0_3)
        window['Y'].update(Y0_3)
        window['Z'].update(Z0_3)

    if event == 'Jacobian Matrix (J)' :
        
        # Defining the equations

        IM = [[1,0,0],[0,1,0],[0,0,1]]
        i = [[0],[0],[1]]
        d0_3 = H0_3[0:3,3:]

        # Row 1 - 3 column 1
        J1a = (np.dot(IM,i))

        # Cross product of Row 1 - 3 column 1

        J1 = [
            [(J1a[1,0]*d0_3[2,0])-(J1a[2,0]*d0_3[1,0])],
            [(J1a[2,0]*d0_3[0,0])-(J1a[0,0]*d0_3[2,0])],
            [(J1a[0,0]*d0_3[1,0])-(J1a[1,0]*d0_3[0,0])]
            ]

        # Row 1 - 3 column 2
        R0_1a = np.dot(H0_1,1)
        R0_1b = R0_1a[0:3, 0:3]
        d0_1 = R0_1a[0:3,3:]
        J2a = (np.dot(R0_1b,i))
        J2b = (np.subtract(d0_3,d0_1))

        # Cross product of Row 1 - 3 column 2

        J2 = [
            [(J2a[1,0]*J2b[2,0])-(J2a[2,0]*J2b[1,0])],
            [(J2a[2,0]*J2b[0,0])-(J2a[0,0]*J2b[2,0])],
            [(J2a[0,0]*J2b[1,0])-(J2a[1,0]*J2b[0,0])]
            ]

        # Row 1 - 3 column 3
        R0_2 = H0_2[0:3,0:3]
        J3 = (np.dot(R0_2,i))
        J3a = [[0], [0], [0]]

        # Jacobian Matrix
        JM1 = np.concatenate((J1, J2, J3), 1)
        JM2 = np.concatenate((J1a, J2a, J3a), 1)
        Jacobian = np.concatenate((JM1, JM2), 0)
        sg.popup('J =', Jacobian)

        # Disabler program 
        disable_J.update(disabled=True)
        disable_D.update(disabled=False)
        disable_IV.update(disabled=False)
        disable_TJ.update(disabled=False)

    if event == 'Det(J)' :
        DJ = np.linalg.det(JM1)
        DJ = np.around(DJ, 4)
        #print("D(J) = ", DJ)
        sg.popup('D(J) = ', DJ)

        if 0.0 >= DJ > -1.0:
            disable_IV.update(disabled=True)
            sg.popup('Warning: This is Non-Invertible')

        elif DJ != 0.0 or DJ != -0.0:
            disable_IV.update(disabled=False)
    
    if event == 'Inverse of J' :
        IJ = np.linalg.inv(JM1)
        sg.popup('I(J) = ', IJ)

    if event == 'Transpose of J' :
        TJ = np.transpose(JM1)
        sg.popup('T(J) = ', TJ)

    if event == 'Submit' :
        df = df.append(values, ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        sg.popup('Data Saved!')

    if event == 'Inverse Kinematics':
        inverse_kinematics_window()

window.close()
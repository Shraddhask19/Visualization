from tkinter import *
from tkinter import filedialog
import pandas as pd
import math
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import operator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import itertools
import numpy as np
from sklearn.tree import _tree
from sklearn import metrics
from tkinter import messagebox as mb
from sklearn.neighbors import KNeighborsClassifier

    
def browseDataset():
    filename = filedialog.askopenfilename(initialdir="/",title="Select dataset", filetypes=(("CSV files", "*.csv*"), ("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: "+filename)
    newfilename = ''
    print(filename)
    for i in filename:
        if i == "/":
            newfilename = newfilename + "/"
        newfilename = newfilename + i
    print(newfilename)
    data = pd.read_csv(filename)
    d = pd.read_csv(filename)
    w = Tk()
    w.title("2019BTECS00025-Data Analysis Tool")
    w.geometry("600x500")
    
    tv1 = ttk.Treeview(w)
    tv1.place(relheight=1, relwidth=1)

    treescrolly = Scrollbar(w, orient="vertical", command=tv1.yview) 
    treescrollx = Scrollbar(w, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y") 
    tv1["column"] = list(data.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) 

    df_rows = data.to_numpy().tolist() 
    for row in df_rows:
        tv1.insert("", "end", values=row)
    
    # Creating Menubar
    menubar = Menu(w)
    
    # Adding File Menu and commands
    assignements = Menu(menubar, tearoff = 0)
    menubar.add_cascade(label ='Assignment', menu = assignements)
    assignements.add_command(label ='Assignment 1', command = lambda: GoToAssignment("Assignment1"))
    assignements.add_command(label ='Assignment 2', command = lambda: GoToAssignment("Assignment2"))
    assignements.add_command(label ='Assignment 3', command = lambda: GoToAssignment("Assignment3"))
    assignements.add_command(label ='Assignment 4', command = lambda: GoToAssignment("Assignment4"))
    assignements.add_command(label ='Assignment 5', command = lambda: GoToAssignment("Assignment5"))
    
    # display Menu
    w.config(menu = menubar,bg='#18253f')
    
    def GoToAssignment(assignment):          
        if assignment == "Assignment1":
            window1 = Tk()
            window1.title("Assignment1")
            window1.geometry("300x300")
            menubar = Menu(window1)
            questions = Menu(menubar, tearoff = 0)
            menubar.add_cascade(label ='Topics', menu = questions)
            questions.add_command(label ='Data Display', command = lambda: SolveQuestion("Data Display"))
            questions.add_command(label ='Measure of central tendencies', command = lambda: SolveQuestion("Measure of central tendencies"))
            questions.add_command(label ='Dispersion of data', command = lambda: SolveQuestion("Dispersion of data"))
            questions.add_command(label ='Plots', command = lambda: SolveQuestion("Plots"))
            Label(window1,text="Select Topic from Menu", font=('Verdana', 14), fg="#fff",bg="#555",height=4).grid(row=0,column=0,padx=20,pady=30)
            def SolveQuestion(question):
                if question == "Data Display":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    tv1 = ttk.Treeview(window2)
                    tv1.place(relheight=1, relwidth=1)

                    treescrolly = Scrollbar(w, orient="vertical", command=tv1.yview) 
                    treescrollx = Scrollbar(w, orient="horizontal", command=tv1.xview)
                    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
                    treescrollx.pack(side="bottom", fill="x")
                    treescrolly.pack(side="right", fill="y") 
                    tv1["column"] = list(data.columns)
                    tv1["show"] = "headings"
                    for column in tv1["columns"]:
                        tv1.heading(column, text=column) 

                    df_rows = data.to_numpy().tolist() 
                    for row in df_rows:
                        tv1.insert("", "end", values=row)
                    
                    window2.mainloop()
                elif question == "Measure of central tendencies":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute = StringVar(window2)
                    clickedAttribute.set("Select Attribute")
                    dropCols = OptionMenu(window2, clickedAttribute, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    measureOfCentralTendancies = ["Mean","Mode","Median","Midrange","Variance","Standard Deviation"]
                    clickedMCT = StringVar(window2)
                    clickedMCT.set("Select MCT")
                    dropMCT = OptionMenu(window2, clickedMCT, *measureOfCentralTendancies)
                    dropMCT.grid(column=2,row=5)
                    Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30)

                    def computeOperation():
                        attribute = clickedAttribute.get()
                        operation = clickedMCT.get()
                        if operation == "Mean":
                            sum = 0
                            for i in range(len(data)):
                                sum += data.loc[i, attribute]
                            avg = sum/len(data)
                            res = "Mean of given dataset is ("+attribute+") "+str(avg)
                            Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                        elif operation == "Mode": 
                            freq = {}
                            for i in range(len(data)):
                                freq[data.loc[i, attribute]] = 0
                            maxFreq = 0
                            maxFreqElem = 0
                            for i in range(len(data)):
                                freq[data.loc[i, attribute]] = freq[data.loc[i, attribute]]+1
                                if freq[data.loc[i, attribute]] > maxFreq:
                                    maxFreq = freq[data.loc[i, attribute]]
                                    maxFreqElem = data.loc[i, attribute]
                            res = "Mode of given dataset is ("+attribute+") "+str(maxFreqElem)
                            Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                        elif operation == "Median":
                            n = len(data)
                            i = int(n/2)
                            j = int((n/2)-1)
                            arr = []
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute])
                            arr.sort()
                            if n%2 == 1:
                                res = "Median of given dataset is ("+attribute+") "+str(arr[i])
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                            else:
                                res = "Median of given dataset is ("+attribute+") "+str((arr[i]+arr[j])/2)
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                        elif operation == "Midrange":
                            n = len(data)
                            arr = []
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute])
                            arr.sort()
                            res = "Midrange of given dataset is ("+attribute+") "+str((arr[n-1]+arr[0])/2)
                            Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                        elif operation == "Variance" or operation == "Standard Deviation":
                            sum = 0
                            for i in range(len(data)):
                                sum += data.loc[i, attribute]
                            avg = sum/len(data)
                            sum = 0
                            for i in range(len(data)):
                                sum += (data.loc[i, attribute]-avg)*(data.loc[i, attribute]-avg)
                            var = sum/(len(data))
                            if operation == "Variance":
                                res = "Variance of given dataset is ("+attribute+") "+str(var)
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                            else:
                                res = "Standard Deviation of given dataset is ("+attribute+") "+str(math.sqrt(var)) 
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)  
                    window2.mainloop()
                elif question == "Dispersion of data":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute = StringVar(window2)
                    clickedAttribute.set("Select Attribute")
                    dropCols = OptionMenu(window2, clickedAttribute, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    dispersionOfData = ["Range","Quartiles","Inetrquartile range","Minimum","Maximum"]
                    clickedDispersion = StringVar(window2)
                    clickedDispersion.set("Select Dispersion Operation")
                    dropDisp = OptionMenu(window2, clickedDispersion, *dispersionOfData)
                    dropDisp.grid(column=2,row=5)
                    Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30)
                    
                    def computeOperation():
                        attribute = clickedAttribute.get()
                        operation = clickedDispersion.get()
                        if operation == "Range":
                            arr = []
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute])
                            arr.sort()
                            res = "Range of given dataset is ("+attribute+") "+str(arr[len(data)-1]-arr[0])
                            Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                        elif operation == "Quartiles" or operation == "Inetrquartile range": 
                            arr = []
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute])
                            arr.sort()
                            if operation == "Quartiles": 
                                res1 = "Lower quartile(Q1) is ("+attribute+") "+str(arr[int((len(arr)+1)/4)])
                                res2 = "Middle quartile(Q2) is ("+attribute+") "+str(arr[int((len(arr)+1)/2)])
                                res3 = "Upper quartile(Q3) is ("+attribute+") "+str(arr[int(3*(len(arr)+1)/4)])
                                Label(window2,text=res1,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                                Label(window2,text=res2,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=8)
                                Label(window2,text=res3,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=9)
                            else:
                                res = "Interquartile range(Q3-Q1) of given dataset is ("+attribute+") "+str(arr[int(3*(len(arr)+1)/4)]-arr[int((len(arr)+1)/4)])
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=8)
                                
                        elif operation == "Minimum" or operation == "Maximum":
                            arr = []
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute])
                            arr.sort()
                            if operation == "Minimum":
                                res = "Minimum value of given dataset is ("+attribute+") "+str(arr[0])
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                            else:
                                res = "Maximum value of given dataset is ("+attribute+") "+str(arr[len(data)-1])
                                Label(window2,text=res,width=80,height=3,fg='green',font=('Verdana', 12)).grid(column=1,row=7)
                    window2.mainloop()
                elif question == "Plots":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute1 = StringVar(window2)
                    clickedAttribute1.set("Select Attribute 1")
                    clickedAttribute2 = StringVar(window2)
                    clickedAttribute2.set("Select Attribute 2")
                    clickedClass = StringVar(window2)
                    clickedClass.set("Select class")
                    plots = ["Quantile-Quantile Plot","Histogram","Scatter Plot","Boxplot"]
                    clickedPlot = StringVar(window2)
                    clickedPlot.set("Select Plot")
                    dropPlots = OptionMenu(window2, clickedPlot, *plots)
                    dropPlots.grid(column=1,row=6,padx=20,pady=30)
                    Button(window2,text="Select Attributes",command= lambda:selectAttributes()).grid(column=2,row=6,padx=20,pady=30)
                    
                    def computeOperation():
                        attribute1 = clickedAttribute1.get()
                        attribute2 = clickedAttribute2.get()
                        
                        operation = clickedPlot.get()
                        if operation == "Quantile-Quantile Plot": 
                            arr = []
                            sum = 0
                            for i in range(len(data)):
                                arr.append(data.loc[i, attribute1])  
                                sum += data.loc[i, attribute1]
                            avg = sum/len(arr)
                            sum = 0
                            for i in range(len(data)):
                                sum += (data.loc[i, attribute1]-avg)*(data.loc[i, attribute1]-avg)
                            var = sum/(len(data))
                            sd = math.sqrt(var)
                            z = (arr-avg)/sd
                            stats.probplot(z, dist="norm", plot=plt)
                            plt.title("Normal Q-Q plot")
                            plt.show()
                            
                        elif operation == "Histogram": 
                            sns.set_style("whitegrid")
                            sns.FacetGrid(data, hue=clickedClass.get(), height=5).map(sns.histplot, attribute1).add_legend()
                            plt.title("Histogram")
                            plt.show(block=True)
                        elif operation == "Scatter Plot":
                            sns.set_style("whitegrid")
                            sns.FacetGrid(data, hue=clickedClass.get(), height=4).map(plt.scatter, attribute1, attribute2).add_legend()
                            plt.title("Scatter plot")
                            plt.show(block=True)
                        elif operation == "Boxplot":
                            sns.set_style("whitegrid")
                            sns.boxplot(x=attribute1,y=attribute2,data=data)
                            plt.title("Boxplot")
                            plt.show(block=True)
                        
                    def selectAttributes():
                        operation = clickedPlot.get()
                        if operation == "Quantile-Quantile Plot":
                            dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                            dropCols.grid(column=3,row=8,padx=20,pady=30)  
                            Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
                        
                        elif operation == "Histogram":   
                            dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                            dropCols.grid(column=3,row=8,padx=20,pady=30)  
                            dropCols = OptionMenu(window2, clickedClass, *cols)
                            dropCols.grid(column=5,row=8,padx=20,pady=30) 
                            Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
                    
                        elif operation == "Scatter Plot":
                            dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                            dropCols.grid(column=2,row=8,padx=20,pady=30)
                            dropCols = OptionMenu(window2, clickedAttribute2, *cols)
                            dropCols.grid(column=3,row=8,padx=20,pady=30)
                            dropCols = OptionMenu(window2, clickedClass, *cols)
                            dropCols.grid(column=5,row=8)
                            Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)

                        elif operation == "Boxplot":
                            dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                            dropCols.grid(column=2,row=8,padx=20,pady=30)
                            dropCols = OptionMenu(window2, clickedAttribute2, *cols)
                            dropCols.grid(column=3,row=8,padx=20,pady=30)
                            Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
                    window2.mainloop()
            window1.config(menu = menubar)
            window1.mainloop()
        
        elif assignment == "Assignment2":
            window1 = Tk()
            window1.title("Assignment2")
            window1.geometry("300x300")
            menubar = Menu(window1)
            questions = Menu(menubar, tearoff = 0)
            menubar.add_cascade(label ='Topics', menu = questions)
            questions.add_command(label ='Chi-Square Test', command = lambda: SolveQuestion("Chi-Square Test"))
            questions.add_command(label ='Correlation(Pearson) Coefficient', command = lambda: SolveQuestion("Correlation(Pearson) Coefficient"))
            questions.add_command(label ='Normalization Techniques', command = lambda: SolveQuestion("Normalization Techniques"))
            Label(window1,text="Select Topic from Menu", font=('Verdana', 14), fg="#fff",bg="#555",height=4).grid(row=0,column=0,padx=20,pady=30)
            def SolveQuestion(question):
                if question == "Chi-Square Test":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute1 = StringVar(window2)
                    clickedAttribute1.set("Select Attribute1")
                    dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    clickedAttribute2 = StringVar(window2)
                    clickedAttribute2.set("Select Attribute2")
                    dropCols = OptionMenu(window2, clickedAttribute2, *cols)
                    dropCols.grid(column=2,row=5)
                    clickedClass = StringVar(window2)
                    clickedClass.set("Select Class")
                    dropCols = OptionMenu(window2, clickedClass, *cols)
                    dropCols.grid(column=3,row=5)
                    Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30) 
                    
                    def computeOperation():
                        attribute1 = clickedAttribute1.get()
                        attribute2 = clickedAttribute2.get()
                        category = clickedClass.get()
                        arrClass = data[category].unique()
                        g = data.groupby(category)
                        f = {
                        attribute1: 'sum',
                        attribute2: 'sum'
                        }
                        v1 = g.agg(f)
                        print(v1)
                        v = v1.transpose()
                        print(v)
                        
                        tv1 = ttk.Treeview(window2,height=3)
                        tv1.grid(column=1,row=8,padx=5,pady=8)
                        tv1["column"] = list(v.columns)
                        tv1["show"] = "headings"
                        for column in tv1["columns"]:
                            tv1.heading(column, text=column)

                        df_rows = v.to_numpy().tolist()
                        for row in df_rows:
                            tv1.insert("", "end", values=row)

                        total = v1[attribute1].sum()+v1[attribute2].sum()
                        chiSquare = 0.0
                        for i in arrClass:
                            chiSquare += (v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))*(v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))/(((v[i].sum())*(v1[attribute1].sum()))/total)
                            chiSquare += (v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))*(v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))/(((v[i].sum())*(v1[attribute2].sum()))/total)
                        
                        degreeOfFreedom = (len(v)-1)*(len(v1)-1)
                        Label(window2,text="Chi-square value is "+str(chiSquare), justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=9,padx=5,pady=8) 
                        Label(window2,text="Degree of Freedom is "+str(degreeOfFreedom), justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=10,padx=5,pady=8) 
                        res = ""
                        if chiSquare > degreeOfFreedom:
                            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are strongly correlated."
                        else:
                            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are not correlated."
                        Label(window2,text=res, justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=11,padx=5,pady=8)
                    window2.mainloop()
                elif question == "Correlation(Pearson) Coefficient":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute1 = StringVar(window2)
                    clickedAttribute1.set("Select Attribute1")
                    dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    clickedAttribute2 = StringVar(window2)
                    clickedAttribute2.set("Select Attribute2")
                    dropCols = OptionMenu(window2, clickedAttribute2, *cols)
                    dropCols.grid(column=2,row=5)
                    Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30) 
                    
                    def computeOperation():
                        attribute1 = clickedAttribute1.get()
                        attribute2 = clickedAttribute2.get()
                        
                        sum = 0
                        for i in range(len(data)):
                            sum += data.loc[i, attribute1]
                        avg1 = sum/len(data)
                        sum = 0
                        for i in range(len(data)):
                            sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
                        var1 = sum/(len(data))
                        sd1 = math.sqrt(var1)
                        
                        sum = 0
                        for i in range(len(data)):
                            sum += data.loc[i, attribute2]
                        avg2 = sum/len(data)
                        sum = 0
                        for i in range(len(data)):
                            sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
                        var2 = sum/(len(data))
                        sd2 = math.sqrt(var2)
                        
                        sum = 0
                        for i in range(len(data)):
                            sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute2]-avg2)
                        covariance = sum/len(data)
                        pearsonCoeff = covariance/(sd1*sd2)    
                        Label(window2,text="Covariance value is "+str(covariance), justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=8,padx=5,pady=8) 
                        Label(window2,text="Correlation coefficient(Pearson coefficient) is "+str(pearsonCoeff), justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=9,padx=5,pady=8) 
                        res = ""
                        if pearsonCoeff > 0:
                            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are positively correlated."
                        elif pearsonCoeff < 0:
                            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are negatively correlated."
                        elif pearsonCoeff == 0:
                            res = "Attributes " + attribute1 + ' and ' + attribute2 + " are independant."
                        Label(window2,text=res, justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=11,padx=5,pady=8)
                    window2.mainloop()
                elif question == "Normalization Techniques":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute1 = StringVar(window2)
                    clickedAttribute1.set("Select Attribute1")
                    dropCols = OptionMenu(window2, clickedAttribute1, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    clickedAttribute2 = StringVar(window2)
                    clickedAttribute2.set("Select Attribute2")
                    dropCols = OptionMenu(window2, clickedAttribute2, *cols)
                    dropCols.grid(column=2,row=5)
                    clickedClass = StringVar(window2)
                    clickedClass.set("Select class")
                    dropCols = OptionMenu(window2, clickedClass, *cols)
                    dropCols.grid(column=3,row=5)
                    normalizationOperations = ["Min-Max normalization","Z-Score normalization","Normalization by decimal scaling"]
                    clickedOperation = StringVar(window2)
                    clickedOperation.set("Select Normalization Operation")
                    dropOperations = OptionMenu(window2, clickedOperation, *normalizationOperations)
                    dropOperations.grid(column=4,row=5)
                    Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30) 
                    
                    def computeOperation():
                        attribute1 = clickedAttribute1.get()
                        attribute2 = clickedAttribute2.get() 
                        operation = clickedOperation.get()
                        if operation == "Min-Max normalization":
                            n = len(data)
                            arr1 = []
                            for i in range(len(data)):
                                arr1.append(data.loc[i, attribute1])
                            arr1.sort()
                            min1 = arr1[0]
                            max1 = arr1[n-1]
                            
                            arr2 = []
                            for i in range(len(data)):
                                arr2.append(data.loc[i, attribute2])
                            arr2.sort()
                            min2 = arr2[0]
                            max2 = arr2[n-1]
                            
                            for i in range(len(data)):
                                d.loc[i, attribute1] = ((data.loc[i, attribute1]-min1)/(max1-min1))
                            
                            for i in range(len(data)):
                                d.loc[i, attribute2] = ((data.loc[i, attribute2]-min2)/(max2-min2))
                        elif operation == "Z-Score normalization":
                            sum = 0
                            for i in range(len(data)):
                                sum += data.loc[i, attribute1]
                            avg1 = sum/len(data)
                            sum = 0
                            for i in range(len(data)):
                                sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
                            var1 = sum/(len(data))
                            sd1 = math.sqrt(var1)
                            
                            sum = 0
                            for i in range(len(data)):
                                sum += data.loc[i, attribute2]
                            avg2 = sum/len(data)
                            sum = 0
                            for i in range(len(data)):
                                sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
                            var2 = sum/(len(data))
                            sd2 = math.sqrt(var2)
                            
                            for i in range(len(data)):
                                d.loc[i, attribute1] = ((data.loc[i, attribute1]-avg1)/sd1)
                            
                            for i in range(len(data)):
                                d.loc[i, attribute2] = ((data.loc[i, attribute2]-avg2)/sd2)
                        elif operation == "Normalization by decimal scaling":        
                            j1 = 0
                            j2 = 0
                            n = len(data)
                            arr1 = []
                            for i in range(len(data)):
                                arr1.append(data.loc[i, attribute1])
                            arr1.sort()
                            max1 = arr1[n-1]
                            
                            arr2 = []
                            for i in range(len(data)):
                                arr2.append(data.loc[i, attribute2])
                            arr2.sort()
                            max2 = arr2[n-1]
                            
                            while max1 > 1:
                                max1 /= 10
                                j1 += 1
                            while max2 > 1:
                                max2 /= 10
                                j2 += 1
                            
                            for i in range(len(data)):
                                d.loc[i, attribute1] = ((data.loc[i, attribute1])/(pow(10,j1)))
                            
                            for i in range(len(data)):
                                d.loc[i, attribute2] = ((data.loc[i, attribute2])/(pow(10,j2)))
                        
                        Label(window2,text="Normalized Attributes", justify='center',height=2,fg='green',font=('Verdana', 12)).grid(column=1,row=8,padx=5,pady=8)         
                        tv1 = ttk.Treeview(window2,height=15)
                        tv1.grid(column=1,row=9,padx=5,pady=8)
                        tv1["column"] = [attribute1,attribute2]
                        tv1["show"] = "headings"
                        for column in tv1["columns"]:
                            tv1.heading(column, text=column)
                        i = 0
                        while i < len(data):
                            tv1.insert("", "end", iid=i, values=(d.loc[i, attribute1],d.loc[i, attribute2]))
                            i += 1
                        sns.set_style("whitegrid")
                        sns.FacetGrid(d, hue=clickedClass.get(), height=4).map(plt.scatter, attribute1, attribute2).add_legend()
                        plt.title("Scatter plot")
                        plt.show(block=True)
                    window2.mainloop()
            window1.config(menu = menubar)
            window1.mainloop()
        
        elif assignment == "Assignment3" or assignment == "Assignment4":
            window1 = Tk()
            window1.title(assignment)
            window1.geometry("300x300")
            menubar = Menu(window1)
            questions = Menu(menubar, tearoff = 0)
            menubar.add_cascade(label ='Topics', menu = questions)
            questions.add_command(label ='Information Gain & Gain Ratio', command = lambda: SolveQuestion("Information Gain & Gain Ratio"))
            questions.add_command(label ='Gini Index', command = lambda: SolveQuestion("Gini Index"))
            Label(window1,text="Select Topic from Menu", font=('Verdana', 14), fg="#fff",bg="#555",height=4).grid(row=0,column=0,padx=20,pady=30)
            def SolveQuestion(question):
                if question == "Information Gain & Gain Ratio":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    print(data)
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute = StringVar(window2)
                    clickedAttribute.set("Select Attribute")
                    dropCols = OptionMenu(window2, clickedAttribute, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    
                    dropAttribute = StringVar(window2)
                    dropAttribute.set("Drop Attribute")
                    dropCols = OptionMenu(window2, dropAttribute, *cols)
                    dropCols.grid(column=2,row=5,padx=20,pady=30)
                    
                    d = {}
                    split_i = {}
                    Button(window2,text="Compute",command= lambda:compute()).grid(column=1,row=6,padx=20,pady=30) 
                    Button(window2,text="Drop Column",command= lambda:dropCol()).grid(column=2,row=6,padx=20,pady=30) 
                    
                    def dropCol():
                        print(dropAttribute.get())
                        cols.remove(dropAttribute.get())
                        
                    def compute():
                        cols.remove(clickedAttribute.get())
                        print(clickedAttribute.get())
                        print(cols)
                        arrClass = data[clickedAttribute.get()].unique()
                        g = data.groupby(clickedAttribute.get())
                        print(arrClass, g)
                        f = {
                        clickedAttribute.get() : 'count'
                        }
                        v = g.agg(f)
                        total = 0
                        for category in arrClass:
                            total += v.transpose()[category]
                            
                        info_d = 0
                        for category in arrClass:
                            info_d += calcEntropy(float(v.transpose()[category]), total)
                            
                        for i in cols:
                            arrAttribute = data[i].unique()
                            g1 = data.groupby(i)
                            print(arrAttribute, i)
                            f1 = {
                            i : 'count'
                            }
                            v1 = g1.agg(f1)
                            
                            total1 = 0
                            for eachValue in arrAttribute:
                                total1 += v1.transpose()[eachValue]
                                
                                    
                            info_d1 = 0
                            split_info = 0
                            for eachValue in arrAttribute:
                                for k in arrClass:
                                    num = 0
                                    for j in range(len(data)):
                                        if data.loc[j, clickedAttribute.get()] == k and data.loc[j, i] == eachValue:
                                            num += 1
                                    info_d1 += calcEntropy(num, float(v1.transpose()[eachValue]))
                                info_d1 *= float(v1.transpose()[eachValue])/total1
                                split_info += calcEntropy(float(v1.transpose()[eachValue]), total1)    
                            d[i] = float(info_d1)
                            split_i[i] = float(split_info)
                        
                        sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
                        print(d)
                        print(sorted_d)
                        columns = ('Attributes', 'Info', 'Gain', 'Gain ratio')
                        tv1 = ttk.Treeview(window2, columns=columns, show='headings')
                        tv1.grid(column=1,row=8,padx=5,pady=8)
                        
                        tv1.heading('Attributes', text='Attributes')
                        tv1.heading('Info', text='Info')
                        tv1.heading('Gain', text='Gain')
                        tv1.heading('Gain ratio', text='Gain ratio')
                        
                        tuples = []
                        for n in sorted_d:
                            tuples.append((f'{n}', f'{sorted_d[n]}', f'{float(info_d-sorted_d[n])}', f'{(float(info_d-sorted_d[n])/float(split_i[n]))}'))
                        for tuple in tuples:
                            tv1.insert('', END, values=tuple)
                        tv1.grid(row=7, column=1, sticky='nsew')
                        
                        
                        f_names = []
                        c_names = []
                        f_names = cols
                        print(f_names)
                        for c in arrClass:
                            c_names.append(str(c))
                        print(type(c_names))
                        le_class = LabelEncoder()
                        df = data
                        df[clickedAttribute.get()] = le_class.fit_transform(df[clickedAttribute.get()])
                        dft = data.drop(clickedAttribute.get(), axis=1)
                        print(dft)
                        X_train, X_test, Y_train, Y_test = train_test_split(dft, df[clickedAttribute.get()], test_size=0.2, random_state=1)
                        clf = DecisionTreeClassifier(max_depth = 3, random_state = 0,criterion="entropy")
                        model = clf.fit(X_train, Y_train)
                        Y_predicted = clf.predict(X_test)
                        Y_test = Y_test.to_numpy()
                        print(type(Y_predicted),type(Y_test))
                        print(Y_predicted, "predicted", len(Y_predicted), Y_test, "Y_test", len(Y_test))
                        c_matrix = confusion_matrix(Y_test,Y_predicted)
                        
                        print(c_matrix)
                        print(classification_report(Y_test,Y_predicted))
                        ax = plt.subplot()
                        sns.heatmap(c_matrix, annot=True, fmt='g', ax=ax)
                        ax.set_xlabel('Predicted labels')
                        ax.set_ylabel('True labels') 
                        ax.set_title('Confusion Matrix')
                        ax.xaxis.set_ticklabels(c_names)
                        ax.yaxis.set_ticklabels(c_names)
                        
                        
                        text_representation = tree.export_text(clf)
                        # print(text_representation)
                        print(f_names,c_names)
                        print(type(f_names),type(c_names))
                        fig = plt.figure(figsize=(25,20))
                        _ = tree.plot_tree(clf, feature_names=f_names, class_names=c_names,filled=True)
                        if assignment == "Assignment4":
                            accuracy = "Accuracy " + str(metrics.accuracy_score(Y_test,Y_predicted))
                            Label(window1, text=accuracy).grid(row=10,column=1,padx=20,pady=5)
                        plt.show()
                        
                        if assignment == "Assignment4":
                            def get_rules(tree, feature_names, class_names):
                                tree_ = tree.tree_
                                feature_name = [
                                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                                    for i in tree_.feature
                                ]

                                paths = []
                                path = []
                                
                                def recurse(node, path, paths):
                                    
                                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                                        name = feature_name[node]
                                        threshold = tree_.threshold[node]
                                        p1, p2 = list(path), list(path)
                                        p1 += [f"({name} <= {np.round(threshold, 3)})"]
                                        recurse(tree_.children_left[node], p1, paths)
                                        p2 += [f"({name} > {np.round(threshold, 3)})"]
                                        recurse(tree_.children_right[node], p2, paths)
                                    else:
                                        path += [(tree_.value[node], tree_.n_node_samples[node])]
                                        paths += [path]
                                        
                                recurse(0, path, paths)

                                # sort by samples count
                                samples_count = [p[-1][1] for p in paths]
                                ii = list(np.argsort(samples_count))
                                paths = [paths[i] for i in reversed(ii)]
                                
                                rules = []
                                for path in paths:
                                    rule = "if "
                                    
                                    for p in path[:-1]:
                                        if rule != "if ":
                                            rule += " and "
                                        rule += str(p)
                                    rule += " then "
                                    if class_names is None:
                                        rule += "response: "+str(np.round(path[-1][0][0][0],3))
                                    else:
                                        classes = path[-1][0][0]
                                        l = np.argmax(classes)
                                        rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                                    rule += f" | based on {path[-1][1]:,} samples"
                                    rules += [rule]
                                    
                                return rules 
                            
                            rules = get_rules(clf, f_names, c_names)
                            win = Tk()
                            win.title("Extracted Rules")
                            win.geometry("500x500")
                            win.config(background="white")
                            i=0
                            for r in rules:
                                Label(win, text=r, justify='center').grid(row=i,column=1,padx=20,pady=5)
                                i=i+1
                            
                            win.mainloop()
                            for r in rules:
                                print(r)
                            
                    def calcEntropy(c,n):
                        if c <= 0:
                            return 0.0 
                        return -(c*1.0/n)*math.log(c*1.0/n, 2)
                    
                    window2.mainloop()
                    
                elif question == "Gini Index":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("500x500")
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    clickedAttribute = StringVar(window2)
                    clickedAttribute.set("Select Attribute")
                    dropCols = OptionMenu(window2, clickedAttribute, *cols)
                    dropCols.grid(column=1,row=5,padx=20,pady=30)
                    
                    dropAttribute = StringVar(window2)
                    dropAttribute.set("Drop Attribute")
                    dropCols = OptionMenu(window2, dropAttribute, *cols)
                    dropCols.grid(column=2,row=5,padx=20,pady=30)
                    
                    d = {}
                    Button(window2,text="Compute",command= lambda:compute()).grid(column=1,row=6,padx=20,pady=30) 
                    Button(window2,text="Drop Column",command= lambda:dropCol()).grid(column=2,row=6,padx=20,pady=30) 
                    
                    def dropCol():
                        print(dropAttribute.get())
                        cols.remove(dropAttribute.get())
                        # window.mainloop()
                        
                    def compute():
                        cols.remove(clickedAttribute.get())
                        print(clickedAttribute.get())
                        print(cols)
                        arrClass = data[clickedAttribute.get()].unique()
                        g = data.groupby(clickedAttribute.get())
                        print(arrClass, g)
                        f = {
                        clickedAttribute.get() : 'count'
                        }
                        v = g.agg(f)
                        total = 0
                        for category in arrClass:
                            total += v.transpose()[category]
                            
                        gini_d = 1
                        for category in arrClass:
                            gini_d -= ((float(v.transpose()[category])/total)*(float(v.transpose()[category])/total))
                            
                        print(gini_d, "gini_d")
                        
                            
                        
                        for i in cols:
                            arrAttribute = data[i].unique()
                            g1 = data.groupby(i)
                            # print(arrAttribute, i)
                            f1 = {
                            i : 'count'
                            }
                            v1 = g1.agg(f1)
                            list1 = []            
                            total1 = 0
                            for eachValue in arrAttribute:
                                total1 += v1.transpose()[eachValue]
                                print(v1.transpose()[eachValue], i)
                                list1.append(eachValue)
                            print(total1, "total1", i) 
                            list_combinations = []
                            for r in range(len(list1)+1):
                                for combination in itertools.combinations(list1, r):
                                    list_combinations.append(combination)
                            
                            list_combinations = list_combinations[1:-1]
                            # prob = 1
                            for t in list_combinations:
                                gini_di = 0
                                for l in t:
                                    prob = 1
                                    for k in arrClass:
                                        num = 0
                                        for j in range(len(data)):
                                            if data.loc[j, clickedAttribute.get()] == k and data.loc[j, i] == l:
                                                num += 1
                                        prob -= ((float(num)/float(v1.transpose()[l]))*(float((num)/float(v1.transpose()[l]))))
                                    print(v1.transpose()[l], total1)
                                    gini_di += ((float(v1.transpose()[l])/float(total1))*float(prob))
                                key = 'Gini '+str(i)+str(t)
                                d[key] = float(gini_di)
                                    
                        dictionary_keys = list(d.keys())
                        sorted_d = {dictionary_keys[i]: sorted(
                            d.values())[i] for i in range(len(dictionary_keys))}
                        
                        columns = ('Criteria', 'Gini Index')
                        tv1 = ttk.Treeview(window2, columns=columns, show='headings')
                        tv1.grid(column=1,row=8,padx=5,pady=8)
                        
                        tv1.heading('Criteria', text='Criteria')
                        tv1.heading('Gini Index', text='Gini Index')
                        
                        tuples = []
                        for n in sorted_d:
                            tuples.append((f'{n}', f'{sorted_d[n]}'))
                        for tuple in tuples:
                            tv1.insert('', END, values=tuple)
                        tv1.grid(row=7, column=1, sticky='nsew')      
                        
                        f_names = []
                        c_names = []
                        f_names = cols
                        print(f_names)
                        for c in arrClass:
                            c_names.append(str(c))
                        print(type(c_names))
                        le_class = LabelEncoder()
                        df = data
                        df[clickedAttribute.get()] = le_class.fit_transform(df[clickedAttribute.get()])
                        dft = data.drop(clickedAttribute.get(), axis=1)
                        print(dft)
                        X_train, X_test, Y_train, Y_test = train_test_split(dft, df[clickedAttribute.get()], test_size=0.2, random_state=1)
                        clf = DecisionTreeClassifier(max_depth = 3, random_state = 0,criterion="gini")
                        model = clf.fit(X_train, Y_train)
                        Y_predicted = clf.predict(X_test)
                        Y_test = Y_test.to_numpy()
                        print(type(Y_predicted),type(Y_test))
                        print(Y_predicted, "predicted", len(Y_predicted), Y_test, "Y_test", len(Y_test))
                        c_matrix = confusion_matrix(Y_test,Y_predicted)
                        
                        print(c_matrix)
                        print(classification_report(Y_test,Y_predicted))
                        ax = plt.subplot()
                        sns.heatmap(c_matrix, annot=True, fmt='g', ax=ax)
                        ax.set_xlabel('Predicted labels')
                        ax.set_ylabel('True labels') 
                        ax.set_title('Confusion Matrix')
                        ax.xaxis.set_ticklabels(c_names)
                        ax.yaxis.set_ticklabels(c_names)
                        
                        
                        text_representation = tree.export_text(clf)
                        # print(text_representation)
                        print(f_names,c_names)
                        print(type(f_names),type(c_names))
                        fig = plt.figure(figsize=(25,20))
                        _ = tree.plot_tree(clf, feature_names=f_names, class_names=c_names,filled=True)
                        if assignment == "Assignment4":
                            accuracy = "Accuracy " + str(metrics.accuracy_score(Y_test,Y_predicted))
                            Label(window1, text=accuracy).grid(row=10,column=1,padx=20,pady=5)
                        plt.show()

                        if assignment == "Assignment4":
                            def get_rules(tree, feature_names, class_names):
                                tree_ = tree.tree_
                                feature_name = [
                                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                                    for i in tree_.feature
                                ]

                                paths = []
                                path = []
                                
                                def recurse(node, path, paths):
                                    
                                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                                        name = feature_name[node]
                                        threshold = tree_.threshold[node]
                                        p1, p2 = list(path), list(path)
                                        p1 += [f"({name} <= {np.round(threshold, 3)})"]
                                        recurse(tree_.children_left[node], p1, paths)
                                        p2 += [f"({name} > {np.round(threshold, 3)})"]
                                        recurse(tree_.children_right[node], p2, paths)
                                    else:
                                        path += [(tree_.value[node], tree_.n_node_samples[node])]
                                        paths += [path]
                                        
                                recurse(0, path, paths)

                                # sort by samples count
                                samples_count = [p[-1][1] for p in paths]
                                ii = list(np.argsort(samples_count))
                                paths = [paths[i] for i in reversed(ii)]
                                
                                rules = []
                                for path in paths:
                                    rule = "if "
                                    
                                    for p in path[:-1]:
                                        if rule != "if ":
                                            rule += " and "
                                        rule += str(p)
                                    rule += " then "
                                    if class_names is None:
                                        rule += "response: "+str(np.round(path[-1][0][0][0],3))
                                    else:
                                        classes = path[-1][0][0]
                                        l = np.argmax(classes)
                                        rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                                    rule += f" | based on {path[-1][1]:,} samples"
                                    rules += [rule]
                                    
                                return rules 
                            
                            rules = get_rules(clf, f_names, c_names)
                            win = Tk()
                            win.title("Extracted Rules")
                            win.geometry("500x500")
                            win.config(background="white")
                            i=0
                            for r in rules:
                                Label(win, text=r, justify='center').grid(row=i,column=1,padx=20,pady=5)
                                i=i+1
                            
                            win.mainloop()
                            for r in rules:
                                print(r)
                        
                    window2.mainloop()
                    
            window1.config(menu = menubar)
            window1.mainloop()
        
        elif assignment == "Assignment5":
            window1 = Tk()
            window1.title(assignment)
            window1.geometry("300x300")
            menubar = Menu(window1)
            questions = Menu(menubar, tearoff = 0)
            menubar.add_cascade(label ='Topics', menu = questions)
            questions.add_command(label ='k-NN Classifier', command = lambda: SolveQuestion("k-NN Classifier"))
            Label(window1,text="Select Topic from Menu", font=('Verdana', 14), fg="#fff",bg="#555",height=4).grid(row=0,column=0,padx=20,pady=30)
            def SolveQuestion(question):
                if question == "k-NN Classifier":
                    window2 = Tk()
                    window2.title(question)
                    window2.geometry("800x500")
                    Label(window2, text="k",font=('Verdana', 12)).grid(row=1,column=1,padx=20,pady=30)
                    answer1 = Entry(window2)
                    answer1.grid(row=1,column=2,padx=20,pady=30)
                    Label(window2, text="Unknown Pattern(Enter comma separated values)",font=('Verdana', 12)).grid(row=2,column=1,padx=20,pady=30)
                    answer2 = Entry(window2)
                    answer2.grid(row=2,column=2,padx=20,pady=30)
                    cols = []
                    for i in data.columns:
                        cols.append(i)
                    Button(window2,text="Compute",font=('Verdana', 12), width=15, height=5,command= lambda:findClass(int(answer1.get()), answer2.get())).grid(column=1,row=6,padx=20,pady=30) 
                    def findClass(k,unknownPattern):
                        ls = unknownPattern.split(",")
                        targetLS = []
                        
                        for s in ls:
                            targetLS.append(float(s))
                        
                        le_class = LabelEncoder()
                        df = data
                        print(data.iloc[:,-1] )
                        df.iloc[: , -1] = le_class.fit_transform(df.iloc[: , -1])
                        print(data.iloc[:,-1] )
                        labelClasses = {}
                        for i in range(len(df)):
                            print(data.iloc[i,-1], type(d.iloc[i,-1]))
                            labelClasses[df.iloc[i,-1]] = d.iloc[i,-1]
                        print(labelClasses)
                        dft = data.iloc[: , :-1] #dropping target column
                        
                        # manual classification

                        uP = tuple(targetLS)
                        allClassPoints = {}
                        for i in range(len(dft)):
                            value = allClassPoints.get(df.iloc[i,-1])
                            if value == None:
                                allClassPoints[df.iloc[i,-1]] = []
                            allClassPoints[df.iloc[i,-1]].append(tuple(dft.iloc[i]))
                        distArray = []
                        for eachClass in allClassPoints:
                            for eachTuple in allClassPoints[eachClass]:
                                sum = 0
                                t = 0
                                while t < (len(eachTuple)):
                                    sum += ((eachTuple[t]-uP[t])**2)
                                    t += 1
                                euclideanDistance = math.sqrt(sum)
                                distArray.append((euclideanDistance,eachClass))
                        
                        distFrequencyDict = {} 
                        maxFreqClass = 0      
                        distArray = sorted(distArray)[:k]
                        for distance in distArray:
                            if distFrequencyDict.get(distance[1]) == None:
                                distFrequencyDict[distance[1]] = 0
                            distFrequencyDict[distance[1]] = distFrequencyDict[distance[1]] + 1
                            if maxFreqClass < distFrequencyDict[distance[1]]:
                                maxFreqClass = distance[1]
                        print(labelClasses[maxFreqClass],"Class Manual")
                        
                        # using in built fn classifier
                        targetLS = [targetLS]
                        X_train, X_test, y_train, y_test = train_test_split(
                        dft, df.iloc[: , -1], test_size = 0.2, random_state=42)
            
                        knn = KNeighborsClassifier(k)
            
                        knn.fit(X_train, y_train)
                        u_pattern = pd.DataFrame(targetLS, columns=list(dft.columns))
                        print(u_pattern,"u_pattern\n")
                        # Predict on dataset which model has not seen before
                        print(knn.predict(u_pattern),type(knn.predict(u_pattern)), "class")

                        Label(window2, text="Predicted Class manually",fg='green',font=('Verdana', 12)).grid(row=7,column=1,padx=20,pady=30)
                        Label(window2, text=labelClasses[maxFreqClass],bg='green',fg='#fff',font=('Verdana', 12)).grid(row=7,column=2,padx=20,pady=30)
                        Label(window2, text="Predicted Class using in built function",fg='green',font=('Verdana', 12)).grid(row=8,column=1,padx=20,pady=30)
                        Label(window2, text=labelClasses[knn.predict(u_pattern)[0]],bg='green',fg='#fff',font=('Verdana', 12)).grid(row=8,column=2,padx=20,pady=30)
                        
                    window2.mainloop()
            window1.config(menu = menubar)
            window1.mainloop()
    
    w.mainloop()                  
        
def Usage():
    mb.showinfo("Product Information", "1.Browse dataset .csv file from file explorer \n2.First select assignment number from menu dropdown \n3.Perform data analysis of your choice from menu\n")


window = Tk()
window.title("2019BTECS00025-Data Analysis Tool")
window.geometry("600x500")
Label(window,text="Data Analysis Tool",justify='center',font=("Verdana", 34),background='#c345fc',foreground='#fff').grid(row=1,column=1,padx=15,pady=20)
label_file_explorer = Label(window,text="Choose Dataset from File Explorer",justify='center',font=("Verdana", 14),height=4,fg="blue")
button_explore = Button(window,text="Browse Dataset", justify='center', width=20, height=4, font=("Verdana", 8),command=browseDataset)
label_file_explorer.grid(column=1,row=2,padx=20,pady=30)
button_explore.grid(column=1,row=3,padx=20,pady=30)
# display Menu
menubar = Menu(window)
helps = Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Usage', menu = helps)
helps.add_command(label ='HowToUse?', command = Usage)
window.config(menu = menubar,bg='#18253f')
window.config(bg='#18253f')
window.mainloop()
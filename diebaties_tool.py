import pandas as pd 
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/storage/emulated/0/Download/modified_datasate.csv")
print(df)

x = df[["AGE","Urea","Cr","HbA1c","Chol","TG","HDL","LDL","VLDL","BMI"]]
y = df[["CLASS_Encoded"]]

model = LogisticRegression()
model.fit(x, y)

age = float(input("enter your age:"))
urea = float(input("enter your urea:"))
cr = float(input("enter your cr:"))
hba1c = float(input("enter your hba1c:"))
chol = float(input("enter your chol:"))
tg = float(input("enter your tg:"))
hdl = float(input("enter your hdl:"))
ldl = float(input("enter your ldl:"))
vldl = float(input("enter your vldl:"))
bmi = float(input("enter your bmi:"))

result = model.predict([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])
print(result)

if result == 0 :
	print("non diabetic")
elif result == 1 :
	print("nondiebetic")
elif result == 2 :
	print("prediebetic")
elif result == 3 :
	print("diebetic")
elif result == 4:
    print("diebetic")
    
	
else:
	print("other class output")
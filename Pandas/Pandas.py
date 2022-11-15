import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
#load data into a DataFrame object:
df = pd.DataFrame(data)
print(df) 

# Use loc to return one or more specified rows
print(df.loc[0])
print(df.loc[[0, 1]]) # để trong [] kết quả sẽ là DataFrame


# Pandas read CSV ( common seperate files )
# Load CSV into a DataFrame, read_csv() ; to_string(): print entire Dataframe
import pandas as pd
df = pd.read_csv('data.csv')
print(df.to_string()) # nếu không có to_string(), thì kqua sẽ là 5 dòng đầu và 5 dòng cuối.

# Max_rows
# Kiểm tra system's maximun rows
import pandas as pd
print(pd.options.display.max_rows) 
pd.options.display.max_rows = 9999 # đặt số dòng max 



# Pandas read Json
import pandas as pd
df = pd.read_json('data.json')
print(df.to_string()) 


# head(), tail(). info()


# Pandas - cleaning empty cells
# Use dropna() trả về 1 df mới mà no empty nhu 1 copy
# Nếu mà thay luôn df cũ thì thêm inplace = True
import pandas as pd
df = pd.read_csv('data.csv')
df.dropna(inplace = True)
print(df.to_string())

# Replace empty values
# fillna() thay the gia tri null thanh gia tri moi
import pandas as pd
df = pd.read_csv('data.csv')
df.fillna(130, inplace = True)
df.fillna(method = 'backfill', inplace = True)
df.fillna(method = 'ffill', inplace = True)

# Replace using mean, median, or mode
import pandas as pd
df = pd.read_csv('data.csv')
x = df["Calories"].mean()
x = df["Calories"].median()
x = df["Calories"].mode()
df["Calories"].fillna(x, inplace = True)


# Pandas - Cleaning Data of Wrong Format ( remove ỏr fix)
# Ví dụ lỗi về format date. đưa nó vè 1 ham : to_datetime(), rồi xóa đi NaT
import pandas as pd
df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(df.to_string())

# Removing Row
df.dropna(subset=["Date"],inplace = True)


# Pandas - Fixing Wrong Data
# sua gia tri tai mot row bat ki
df.loc[7, 'Duration'] = 45 # loc[row,col]

# Giong nhu excel, su dung loop
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120


# Removing Rows
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)


# Pandas - Removing Duplicates
# tra ve tap hop dang boolean, 1 la True, xuat hien 2 lan False
a = df.duplicated()
print(df[a])


# Pandas Correlation _ corr()
# tinh mối tương quan giữa các cột 
df.corr()




























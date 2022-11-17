# Data Type : Numerical, Categorical, Ordinal
# Numberical : biến rời rạc và biến ngẫu nhiên
# Categorical : dữ liệu không thể đo đếm : Yes/no, T/F
# Ordinal : giống categorical, nh có thể đếm, ví dụ điểm A > điểm B

# Machine Learning with mean, median, mode
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# use numpy for calculating
import numpy as np
x = np.median(speed)
x = np.mean(speed)


# Or use Scipy module for mode()
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = stats.mode(speed)
print(x)

# Standard Deviation and variance
x = np.std(speed)
y = np.var(speed)

# Percentile(obj,%) in numpy
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = np.percentile(ages, 75)
print(x)

# Data Distribution (các loại phân phối dữ liệu.)
import matplotlib.pyplot as plt
x = np.random.uniform(0.0, 5.0, 250)
print(x)
plt.hist(x,5)
plt.show()


# Linear Regression : Hồi quy tuyến tính
# Dựa vào các điểm sẵn có để đưa ra dự đoán dựa trên 1 đường thẳng
import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show()

# Import Scipy to draw line of linear regression
import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

#explanation
slope, intercept, r, p, std_err = stats.linregress(x, y)
# execute a method that returns some important key value òf Linear Regression
# tạo ra 1 hàm dựa trên giá trị của slope and intercept, sẽ trả về giá trị y, với giá trị x đã thay mới
def myfunc(x):
  return slope * x + intercept
# chạy mỗi giá trị x, trả về một chuỗi mới với giá trị y mới
mymodel = list(map(myfunc, x))
# Draw origin scatter plot
plt.scatter(x,y)
# Draw line regression
plt.plot(x, mymodel)

# display diagram
plt.show()


# what ís R for relationship
# The coefficient of correlation - is call r
# the r value from -1 to 1, where 0 is no relationship
# use scipy to compute
print(r)
# Note : result is -0.76 , relationship is not perfect, but we can use linear regression in future predict


# Predict Future Values
# Use myfunc() _ users_function() to predict
# Vi du predict the speed of a 10 years old car
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
speed = myfunc(10)
print(speed)

# What is bad fit ?
# điều này xảy ra khi các điểm phân tách quá lớn, không đồng nhất, 
# dẫn đến linear không có tập trung, và giá trị r tiệp cận đến 0.



# Polynomial Regression
import numpy
import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

# explanation
# numpy có method để tạo nên polynomial model
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# tạo ra giá trị cho x, từ 1 đến 22 với n điểm
myline = numpy.linspace(1, 22, 100)
# Trực quan hóa các giá trị, cũ và mới.
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()



# R-Squared
# Giống như r trong scipy, r-squared cũng biểu thị mối quan hệ của x và y
# giá trị từ 0 đến 1, 0 is no relation, 1 ís 100% relation
# Python and Sklearn có thể tính đc giá trị này
import numpy
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
print(r2_score(y, mymodel(x)))

# Tương tự như bài toán trên, chúng ta sử dụng mymodel 
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# Ví dụ dự đoán tốc độ của xe lúc 17:00
import numpy
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
speed = mymodel(17)
print(speed)



# Bài toán Multiple Regression
# Đọc file từ kho lưu trữ : data.csv
import pandas
df = pandas.read_csv("data.csv")

# Đặt những giá trị độc lập vào biến X
# Đặt những biến phụ thuộc vào biến y
X = df[['Weight', 'Volume']]
y = df['CO2']

# Sử dụng một vài method trong sklearn 
from sklearn import linear_model
# từ sklearn tạo ra 1 object với LinearRegression()
# Object dùng fit() để lấy biến độc lập và biến phụ thuộc như thành phần và fill vào regression object với data đc miêu tả theo relation
regr = linear_model.LinearRegression()
regr.fit(X, y)

# khi đó regr là 1 regression object to predict Co2 values dựa trên weight and volume
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])
# in ra
print(predictedCO2)


# Coeffient
# là giá trị thể hiện sự tương quan giữa các giá trị unknown
# ví dụ 2x, thì x là unkonwn, 2 là coeffient
# Dựa vào model bài trên, ta tính được hệ số ẩn này
print(regr.coef_)
# result is [0.00755095 0.00780526]
# tức là với weight tăng 1kg, thì Co2 tăng lên 0.00780526
# nếu volume tăng lên 1cm3, thì co2 tăng lên 0.00780526




# Machine Learning _ Scale
# When your data has different values, and even different measurement units, it can be difficult to compare them. What is kilograms compared to meters? Or altitude compared to time?

#The answer to this problem is scaling. We can scale data into new values that are easier to compare.

# Sử dụng Standardization method
# z = (x-u)/s 
# trong đó x ( origin value ), u(mean), s(std)
# Ví dụ cột weight, tính mean and std, rồi chuẩn hóa theo z

# Trong sklearn có 1 hàm để làm : StandardScaler() trả về một scaler object với việc chuyển hóa data
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df = pandas.read_csv("data.csv")
X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)
print(scaledX)



# Predict CO2 value
# dự đoán lượng Co2 với 1.3 lít car và 2300 weight kilogram
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df = pandas.read_csv("data.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']
scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)



# Machine Learning - Train/Test
# Start with data set
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
plt.scatter(x, y)
plt.show()

# truc x : số phút trước khi mua hàng
# trục y : lượng tiền bỏ ra khi mua

# Split Into Train/ Test
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

# Display the training set
plt.scatter(train_x, train_y)
plt.show()
# Display the testing set
plt.scatter(test_x, test_y)
plt.show()

# Fit the Data Set
# Dự đoán, nên sử dụng polinomial regression
# dùng plot() trong matplotlib để thể hiện. 
import numpy
import matplotlib.pyplot as plt
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0, 6, 100)
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# kiểm tra độ overfitting , ở phút thứ 6, gtri trả về có vẻ hơi sai
# Khi đó, sử dụng R-squared để kiểm tra độ fitting của model
# R2 ís also r-squared
# sklearn có 1 module r2_score() 
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2 = r2_score(train_y, mymodel(train_x))
print(r2)
# Kết quả trả về 0.799 thì có vẻ khá OK

# Bring in the Testing set. cải thiện độ chính xác
import numpy
from sklearn.metrics import r2_score
numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2 = r2_score(test_y, mymodel(test_x))
print(r2)

# Bên testing, kqua là 0.809, điều đó chứng tỏ model fit tốt. Và chúng ta có thể kthuc

# Predict values
print(mymodel(5))




# Tree Decision
# là 1 flow chart để giúp đưa ra các quyết định dưạ trên các trải nghiêm quá khứ
# Đọc data set với pandas
import pandas
df = pandas.read_csv("data.csv")
print(df)

# Để đưa về a decision tree, tất cả dữ liệu phải là numerical
# Vì thế, phải convert dữ liệu ở Nationality và Go thành numerical
# Pandas có hàm map() để chuyển đổi dữ liệu
# {'UK': 0, 'USA': 1, 'N': 2}
# Chuyển strings thành numerical values
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
print(df)

# Tách df thành các mảng cho X và y
# Với X là features columns, y là target columns
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
print(X)
print(y)

# Tạo và display a Decision Tree
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
df = pandas.read_csv("data.csv")
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
tree.plot_tree(dtree, feature_names=features)

print(dtree.predict([[40,10,7,1]]))

# Explanation
# Rank <= 6.5, tức là mọi event <=6.5 sẽ theo nhánh True, ngược lại theo nhánh False
# gini = 0.497 ( 0 to 0.5 )if 0 is the same, 0.5 thì nó là middle
# sample = 13 số sample đầu vào
# value =[6,7] 6 get No, 7 get Go
# Gini = 1-(x/n)^2 -(y/n)^2
# 1-(7/13)^2 - (6/13)^2 = 0.497 , chỉ có 49,7% là đi theo một chiều. 


# Predict Values
# Should I go see a show starring a 40 years old American comedian, with 10 years of experience, and a comedy ranking of 7?
# Decision Tree không trả về 1 đáp án giống nhau, dù cùng data, bởi vì nó sẽ phụ thộc vào xác xuất của các outcome




# Creating a confusion matrix
import numpy
# Tạo ra các giá trị ' actual ' and ' predicted' values
actual = numpy.random.binomial(1, 0.9, size = 1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)

# Dùng metrics trong sklearn để tạo ra confusion matrix
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(actual,predicted)

# Tạo 1 table, chứa confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix= confusion_matrix, display_labels=[False,True])

# Visualize the display
import matplotlib.pyplot as plt
cm_display.plot()
plt.show()

# Explanation
# Mô hình matrix sẽ thưcj hiện được rất nhiều measures :
# Accuracy : = (True Positive + True Negative )/ total prediction
Accuracy = metrics.accuracy_score(actual, predicted)

# Precision = True Positive / ( True positive + false positive )
Precision = metrics.precision_score(actual, predicted)

# Sensitivity ( recall ) = True positive /(true positive + false negative)
Sensitivity_recall = metrics.recall_score(actual, predicted)


# Machine Learning - Hierarchical Clustering : Phân cụm theo bậc
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.show()

# Sử dụng sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
plt.scatter(x, y, c=labels)
plt.show()


# Explanation
# Gộp 2 gia tri x,y theo cap : (4,21),.....
data = list(zip(x, y))



# Machine learning - Logistics Regression
# để giải các bài toán liên quan tới classification bằng việc dự đoán categorical outcome
# còn Linear Regression thì dựa trên numberical, continuous outcome
# đưa về 1 col và 1 row thì logistics regression mới hd
import numpy
#X represents the size of a tumor in centimeters.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

from sklearn import linear_model

# tạo ra 1 object , LogisticsRegression()
logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))


# Coefficient : the expected change in log-odds
import numpy
from sklearn import linear_model

#Reshaped for Logistic function.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
logr = linear_model.LogisticRegression()
logr.fit(X,y)
log_odds = logr.coef_
odds = numpy.exp(log_odds)
print(odds) # result : 4.03410341
#This tells us that as the size of a tumor increases by 1mm the odds of it being a tumor increases by 4x.

# Probability : Dùng hàm để tính ra, k sài scipy
def logit2prob(logr,x):
  log_odds = logr.coef_ * x + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

# Function explanation
# Viết công thức tính log_odds
log_odds = logr.coef_ * x + logr.intercept_

# để tìm ra odd, thì cần exp lên để trả về odds
odds = numpy.exp(log_odds)
# Xác xuất
probability = odds / (1 + odds)
# Réults explanation
#3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.

# 2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.

# 2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.



# Machine Learning - Grid Search 
# LogisticsRegression trong sklearn có 1 tham số c : control regularization. 
# tìm ra điểm khác biệt và đặt nó làm best score - grid search
# c cao, thì training data sẽ real hơn

# Using đefault parameters
from sklearn import datasets
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# load logistics model for classifying the iris flowers
from sklearn.linear_model import LogisticRegression
# gia du cho c bang 1
logit = LogisticRegression(max_iter= 10000) # define max_iter = 10000 to train many cases
# after creating the model, we must fit the model data
print(logit.fit(X,y))
# to evaluate the model we run the score method
print(logit.score(X,y))

# default c = 1, achieve score is 0.973, tim xem con gia tri nao tot hon khong

# Implementing Grid Search
# Since the default for c is 1, should we set a range around 1
C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

# dung vong lap de chay tung gia tri cua C
score =[]
for choice in C :
  logit.set_params(C=choice)
  logit.fit(X,y)
  score.append(logit.score(X,y))

print(score)
# Result explanation
# với c nhỏ hơn 1 thì less acuracy, nhưng khi tăng tới 1.75 thì more acuracy.
# không phải cứ tăng c thì sẽ tăng acuracy




# Preprocessing - Categorical Data
# transform string to numberical
import pandas as pd
cars = pd.read_csv('data.csv')
print(cars.to_string())

# One Hot Encoding 
# 1 represent for inclusion and O represent for exclusion
# Sử dụng Pandas, có hàm get_dummies() làm one hot encoding
import pandas as pd
cars = pd.read_csv('data.csv')
ohe_cars = pd.get_dummies(cars[['Car']]) # encode cột Cars
print(ohe_cars.to_string())


# Predict CO2
# Sử dụng data từ volumn and weight để dự đoán CO2
# Sử dung hàm concat()
import pandas
cars = pandas.read_csv('data.csv')
ohe_cars = pandas.get_dummies(cars[["Car"]])

X = pandas.concat([cars[['Volumn',"Weight"]], ohe_cars], axis =1 )
y = cars['CO2']

# import sklearn to create a linear model
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X,y)

##predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300,(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)]])



# Dummifying
# xet 1 columns have 2 colors : red and blue
import pandas as pd
colors = pd.DataFrame({'color': ['blue', 'red']})
print(colors)

# use get_dummies() to encode with drop_first to drop the first results
import pandas as pd
colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
dummies = pd.get_dummies(colors, drop_first=True)
dummies['color'] = colors['color']
print(dummies)



# Machine Learning - K-means
# là unsupervise.d learning, chia dữ liệu thành K cluster bằng cách giảm thấp nhất phương sai của chúng
# Mô hình thuật toán
# Mỗi data sẽ đc gán ngẫu nhiên vào K cluster, sau đó sẽ tính centroid( functionally the center)
# Lặp lại quá trình đo cho dến khi không còn sự thay đổi nào giữa lệnh gán
# K means sẽ yêu cầu giá trị K và số cluster cần nhóm lại
# Mỗi điểm sẽ refer 1 eblow - to estimate

import matplotlib.pyplot as plt
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
plt.scatte(x, y)
plt.show()

#
from sklearn.cluster import KMeans
data = list(zip(x, y))
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
# Based on figure, 2 is seem that 2 is good value for K, we continue train



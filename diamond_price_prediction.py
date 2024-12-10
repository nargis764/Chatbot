from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/Dataset/diamonds.csv")

data.head()
data = data.drop("Unnamed: 0", axis=1)

figure = px.scatter(data_frame=data, x='carat', y='price',
                    size='depth', color='cut', trendline='ols')
figure.show()
display(figure)


colors = {'Ideal': 'red', 'Premium': 'blue',
          'Good': 'green', 'Fair': 'orange', 'Very Good': 'purple'}
plt.figure(figsize=(10, 6))

for cut_type in data['cut'].unique():
    subset = data[data['cut'] == cut_type]
    plt.scatter(subset['carat'], subset['price'],
                s=subset['depth'],
                alpha=0.6,
                label=cut_type,
                color=colors[cut_type])

plt.title('Scatter Plot of Carat vs Price')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.legend(title='Cut')
plt.show()


# Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=data,
    x="carat",
    y="price",
    size="depth",
    hue="cut",
    palette="muted",
    alpha=0.7
)
scatter.set(title="Carat vs Price Scatter Plot",
            xlabel="Carat", ylabel="Price")
plt.legend(title="Cut", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

data["size"] = data["x"] * data["y"] * data["z"]
print(data)

fig = px.box(data, x="cut",
             y="price",
             color="color")
fig.show()

label_encoder = LabelEncoder()
data['cut'] = label_encoder.fit_transform(data['cut'])

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
x = np.array(data[['carat', 'cut', 'size']])
y = np.array(data[['price']])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.10, random_state=42)
model = RandomForestRegressor()
model.fit(x_train, y_train)
a = float(input('Carat Size: '))
b = int(input('Cut Type (Fair: 0, Good: 1, Ideal: 2, Premium: 3, Very Good: 4): '))
c = float(input('Size: '))
features = np.array([[a, b, c]])
print("Predicted Diamond's Price=", model.predict(features))
numeric_data = data.select_dtypes(include=['number'])
correlation = numeric_data.corr()
print(correlation['price'].sort_values(ascending=False))

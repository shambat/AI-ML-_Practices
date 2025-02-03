from fastapi import FastAPI
import uvicorn

# Create FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def main():
    return {"message": "Welcome SHAM!"}

# Name endpoint
@app.get("/{name}")
def hello_name(name: str):
    return {"message": f"Welcome to GeeksforGeeks!, {name}"}

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# Loading Iris Dataset
iris = load_iris()

# Getting features and targets from the dataset
X = iris.data
Y = iris.target

# Fitting our Model on the dataset
clf = GaussianNB()
clf.fit(X,Y)

from pydantic import BaseModel

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post('/predict')
    
def predict(data : request_body):
    test_data = [[
            data.sepal_length, 
            data.sepal_width, 
            data.petal_length, 
            data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}



# Run the server
if __name__ == "__main__":
    uvicorn.run("dep:app", host="127.0.0.1", port=8000, reload=True)

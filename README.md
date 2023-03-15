# Stock Price Forecaster
This demo web application demonstrates the use of LSTM networks to predict stock prices. This web application is developed using Flask as Back-end technology and ReactJS with Vite Build tool as Front-end technology for faster and leaner development.

##### 1. Fork this repository
##### 2. Clone the forked repository

#
# Run Locally
#### 1. Setup the Backend
After you clone the repo, open terminal and go to the `backend` folder.
  ```
  cd backend
  ```

Install virtual environment
  ```
  py -m venv venv
  ```
    
Run the virtual environment
  ```
  venv\Scripts\activate
  ```  
    
Install python dependencies
  ```
  pip install -r requirements.txt
  ```

Run the flask server
  ```
  flask run
  ```

The server should start at `port 5000` 

#### 2. Setup the Frontend
Now, head back to the `frontend` folder
  ```
  cd frontend
  ```
    
Install node modules
  ```
  npm install
  ```

#### 3. Run the development server
Finally, run the development server
  ```
  npm run dev
  ```
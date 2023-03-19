import React, { useState, useEffect } from "react";
import axios from "axios";
import Form from "./components/Form";
import StockChart from "./components/Chart";
import TradingViewWidget from "./components/Chart/TradingView";

const App = () => {
  const [status, setStatus] = useState("Offline");
  const [tickerSymbol, setTickerSymbol] = useState("");
  const [date, setDate] = useState([]);
  const [closePrice, setClosePrice] = useState([]);
  const [testDate, setTestDate] = useState([]);
  const [testClosePrice, setTestClosePrice] = useState([]);
  const [trainDate, setTrainDate] = useState([]);
  const [trainClosePrice, setTrainClosePrice] = useState([]);
  const [predictionDate, setPredictionDate] = useState([]);
  const [predictionClosePrice, setPredictionClosePrice] = useState([]);
  const [trainAccuracy, setTrainAccuracy] = useState([]);
  const [testAccuracy, setTestAccuracy] = useState([]);
  const [trainMSE, setTrainMSE] = useState([]);
  const [testMSE, setTestMSE] = useState([]);
  const [trainRMSE, setTrainRMSE] = useState([]);
  const [testRMSE, setTestRMSE] = useState([]);
  const [ready, setReady] = useState();

  axios.defaults.baseURL = "http://127.0.0.1:5000";

  useEffect(() => {
    async function fetchData() {
      const result = await axios("/status");
      console.log(result.data.status);
      setStatus(result.data.status);
    }
    fetchData();
  }, []);

  const get_ticker = (ticker) => {
    setTickerSymbol(ticker);
  };

  const get_data = (get_data) => {
    setDate(get_data.data.date);
    setClosePrice(get_data.data.close);
    setTrainDate(get_data.train.date);
    setTrainClosePrice(get_data.train.close);
    setTestDate(get_data.test.date);
    setTestClosePrice(get_data.test.close);
    setPredictionDate(get_data.prediction.date);
    setPredictionClosePrice(get_data.prediction.close);
    setTrainAccuracy(get_data.train_accuracy);
    setTestAccuracy(get_data.test_accuracy);
    setTrainMSE(get_data.train_mse);
    setTestMSE(get_data.test_mse);
    setTrainRMSE(get_data.train_rmse);
    setTestRMSE(get_data.test_rmse);
    if (
      date &&
      closePrice &&
      testDate &&
      testClosePrice &&
      trainDate &&
      trainClosePrice &&
      predictionDate &&
      predictionClosePrice != []
    )
      setReady(true);
  };

  return (
    <>
      <div className="flex flex-col items-center justify-center">
        <div className="p-4 mt-4">
          {status == "Online" ? (
            <div className="border-2 border-teal-500 text-teal-500 font-bold rounded-full px-3 py-0 flex justify-center items-center gap-2 top">
              <div className="w-3 h-3 bg-teal-500 flex rounded-full"></div>
              {status}
            </div>
          ) : (
            <div className="border-2 border-red-500 text-red-500 font-bold rounded-full px-3 py-0 flex justify-center items-center gap-2 mt-4">
              <div className="w-3 h-3 bg-red-500 flex rounded-full"></div>
              {status}
            </div>
          )}
        </div>
        <div className="flex flex-col items-center w-full gap-20">
          <div>
            <h1 className="text-4xl my-4 text-slate-600 font-bold text-center">
              Stock Price Predictor
            </h1>
            <Form className="mb-2" ticker={get_ticker} series={get_data} />
          </div>
          <TradingViewWidget />
          {ready && (
          <>
            <div
              id="chart-container"
              className="flex flex-col w-full px-4 md:px-12 lg:px-32"
            >
              <p className="mb-3 font-medium text-slate-600 text-3xl text-center p-4">
                {tickerSymbol} Stock Prediction
              </p>
              <StockChart
                date={date}
                close={closePrice}
                ticker={tickerSymbol}
                trainDate={trainDate}
                trainClose={trainClosePrice}
                testDate={testDate}
                testClose={testClosePrice}
                predictionDate={predictionDate}
                predictionClose={predictionClosePrice}
                series="Trend"
                title="Stock Trend"
              />
            </div>
            <div className="flex flex-col justify-between mb-20 w-full px-4 md:px-12 lg:px-32">
                <p className="text-2xl text-slate-600 font-medium text-center mb-4 p-4">
                  Statistics of the designed LSTM model
                </p>
                <table className="rounded-lg overflow-hidden table-auto min-w-full text-center font-medium">
                  <thead className="border-b bg-slate-600 font-medium text-white dark:border-neutral-500 dark:bg-neutral-900">
                    <tr>
                      <th scope="col" class=" px-6 py-4">Statistics</th>
                      <th scope="col" class=" px-6 py-4">Training</th>
                      <th scope="col" class=" px-6 py-4">Testing</th>
                    </tr>
                  </thead>
                  <tbody className="bg-slate-100">
                    <tr className="border-b dark:border-neutral-500">
                      <td className=" px-6 py-4">Accuracy</td>
                      <td className=" px-6 py-4">{trainAccuracy + " %"}</td>
                      <td className=" px-6 py-4 text-green-700">{testAccuracy + " %"}</td>
                    </tr>
                    <tr className="border-b dark:border-neutral-500">
                      <td className=" px-6 py-4">Mean Squared Error</td>
                      <td className=" px-6 py-4">{trainMSE}</td>
                      <td className=" px-6 py-4 text-green-700">{testMSE}</td>
                    </tr>
                    <tr>
                      <td className=" px-6 py-4">Root Mean Squared Error</td>
                      <td className=" px-6 py-4">{trainRMSE}</td>
                      <td className=" px-6 py-4 text-green-700">{testRMSE}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-slate-400 text-sm mb-8 text-center">
                Fetched from backend server
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
};

export default App;

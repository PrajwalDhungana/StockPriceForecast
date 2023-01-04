import React, { useState, useEffect } from "react";
import axios from "axios";
import Form from "./components/Form";
import StockChart from "./components/Chart";

function App() {
  const [status, setStatus] = useState("Offline");
  const [tickerSymbol, setTickerSymbol] = useState("");
  const [date, setDate] = useState([]);
  const [closePrice, setClosePrice] = useState([]);
  const [ready, setReady] = useState(false);

  axios.defaults.baseURL = "http://127.0.0.1:5000";

  useEffect(() => {
    async function fetchData() {
      const result = await axios("/status");
      setStatus(result.data.status);
    }
    fetchData();
  }, []);

  const get_data = (get_data) => {
    setDate(get_data.data.date);
    setClosePrice(get_data.data.close);
    if (date && closePrice !== []) setReady(true);
  };

  const get_ticker = (ticker) => {
    setTickerSymbol(ticker);
  };

  return (
    <div className="flex flex-col items-center">
      <div className="border-2 border-teal-500 text-teal-500 font-bold rounded-full px-3 py-0 flex justify-center items-center gap-2 mt-4">
        <div className="w-3 h-3 bg-teal-500 flex rounded-full"></div>
        {status}
      </div>
      <h1 className="text-4xl mb-5 text-slate-600 font-bold mt-8">
        Stock Price Predictor
      </h1>
      <Form className="mb-14" series={get_data} ticker={get_ticker} />
      {ready === true ? (
        <>
          <StockChart date={date} close={closePrice} ticker={tickerSymbol} />
          <p className="text-slate-400 text-sm mb-8">
            Fetched from backend server
          </p>
        </>
      ) : (
        ""
      )}
    </div>
  );
}

export default App;

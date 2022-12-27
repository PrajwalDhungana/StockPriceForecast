import React, { useState } from "react";
import axios from 'axios'

export default function Form(props) {
  const [tickerSymbol, setTickerSymbol] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    const data = {tickerSymbol};
    if(tickerSymbol !== ''){
      axios.post("http://127.0.0.1:5000/submit", data)
      .then((response) => {
        console.log(response.data);
      })
      .catch((error) => {
        console.error(error);
      });
    }
    setTickerSymbol("");
  };

  const handleChange = (event) => {
    setTickerSymbol(event.target.value);
  };

  return (
    <div className="form-container p-7">
      <form
        className="flex flex-col gap-3"
        onSubmit={handleSubmit}
        method="POST"
      >
        <input
          className="px-5 py-3"
          type="text"
          name="tickerSymbol"
          placeholder="Enter the ticker Symbol"
          onChange={handleChange}
        />
        <button
          className="bg-slate-500 text-white px-5 py-3 hover:bg-slate-600"
          type="submit"
        >
          Forecast
        </button>
      </form>
    </div>
  );
}

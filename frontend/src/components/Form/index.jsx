import React, { useState } from "react";
import axios from "axios";

export default function Form(props) {
  const [tickerSymbol, setTickerSymbol] = useState("");

  const handleSubmit = (event) => {
    event.preventDefault();

    const ticker = { tickerSymbol };
    if (tickerSymbol !== "") {
      props.ticker(tickerSymbol);
      axios
        .post("http://127.0.0.1:5000/submit", ticker)
        .then((response) => {
          console.log(response.data)
          props.series(response.data);
        })
        .catch((error) => {
          console.error(error);
        });
    }
    setTickerSymbol("");
  };

  const handleChange = (event) => {
    setTickerSymbol(event.target.value.toUpperCase());
  };

  return (
    <div className={`${props.className} form-container p-7`}>
      <form
        className="flex flex-col gap-3"
        onSubmit={handleSubmit}
        method="POST"
      >
        <input
          className="px-5 py-3 uppercase w-[500px] placeholder:capitalize border-2 border-slate-400 outline-none"
          type="text"
          name="tickerSymbol"
          placeholder="Enter the ticker symbol"
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

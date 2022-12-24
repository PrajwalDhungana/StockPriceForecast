import React, { useState } from "react";
import APIService from "../api";

export default function Form(props) {
  const [tickerSymbol, setTickerSymbol] = useState([]);

  const insertTicker = () => {
    APIService.InsertTickerSymobl({ tickerSymbol })
      .then((res) => props.insertedTicker(res))
      .catch((error) => console.log("error", error));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const data = {
      tickerSymbol: this.state.tickerSymbol,
    };
    fetch("http://localhost:5000/insertTicker", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Success:", data);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
    setTickerSymbol("");
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
          onChange={(e) => setTickerSymbol(e.target.value)}
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

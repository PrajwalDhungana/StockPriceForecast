import React, { useState } from "react";
import axios from "axios";
import { DatePicker } from "@mui/x-date-pickers";
import { Button, TextField, CircularProgress } from "@mui/material";

export default function Form(props) {
  const [tickerSymbol, setTickerSymbol] = useState("");

  const ticker = { tickerSymbol };

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log(tickerSymbol)
    if (tickerSymbol !== "") {
      props.ticker(tickerSymbol);
      axios
        .post("http://127.0.0.1:5000/submit", ticker)
        .then((response) => {
          console.log(response.data);
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
        className="flex gap-3"
        onSubmit={handleSubmit}
        method="POST"
      >
        <TextField id="outlined-basic" label="Ticker Symbol" variant="outlined" value={tickerSymbol} onChange={(e) => handleChange(e)} />
        <DatePicker label="Target prediction date" />
        <Button variant="contained" type="submit">Forecast</Button>
      </form>
    </div>
  );
}

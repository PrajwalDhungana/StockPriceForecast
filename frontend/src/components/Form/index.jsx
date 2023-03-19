import React, { useState } from "react";
import axios from "axios";
import { DatePicker } from "@mui/x-date-pickers";
import { Button, TextField, CircularProgress, styled } from "@mui/material";
import dayjs from "dayjs";

export default function Form(props) {
  const [tickerSymbol, setTickerSymbol] = useState("");
  const [targetDate, setTargetDate] = useState();
  const [selectedDate, setSelectedDate] = useState(null);
  const [loading, setLoading] = useState(false);

  const data = { tickerSymbol, targetDate };

  const handleDateChange = (arg) => {
    setSelectedDate(arg);
    const date = new Date(arg.$d);
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const formattedDate = [year, month, day];
    setTargetDate(formattedDate);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (tickerSymbol !== "") {
      setLoading(true);
      props.ticker(tickerSymbol);
      axios
        .post("http://127.0.0.1:5000/submit", data)
        .then((response) => {
          console.log("Response Data : ", response.data);
          props.series(response.data);
          setLoading(!response.data.valid);
        })
        .catch((error) => {
          console.error(error);
        });
    }
    setTickerSymbol("");
  };

  return (
    <div className={`${props.className} form-container p-7`}>
      <form className="flex gap-3" onSubmit={handleSubmit} method="POST">
        <TextField
          id="outlined-input"
          label="Ticker Symbol"
          variant="outlined"
          value={tickerSymbol}
          onChange={(event) => setTickerSymbol(event.target.value.toUpperCase())}
          required
        />
        <DatePicker
          minDate={dayjs(new Date().setDate(new Date().getDate() + 1))}
          maxDate={dayjs("2024-12-31")}
          label="Target prediction date"
          value={selectedDate}
          onChange={handleDateChange}
          required
        />
        <Button variant="contained" type="submit" disabled={loading}>
          {loading && (
            <CircularProgress
              size={20}
              thickness={5}
              sx={{ marginRight: ".75rem" }}
              color={"inherit"}
            />
          )}
          Forecast
        </Button>
      </form>
    </div>
  );
}

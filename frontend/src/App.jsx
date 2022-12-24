import React, { useState, useEffect } from "react";
import axios from "axios";
import Form from "./components/Form";

function App() {
  const [items, setItems] = useState([]);

  axios.defaults.baseURL = "http://127.0.0.1:5000";

  useEffect(() => {
    async function fetchData() {
      const result = await axios("/items");
      setItems(result.data.items);
    }
    fetchData();
  }, []);

  return (
    <div className="bg-slate-200 flex flex-col h-screen items-center justify-center">
      <h1 className="text-6xl mb-5 text-slate-600 font-bold">Items</h1>
      <ul className="list-disc mb-5">
        {items.map((item) => (
          <li className="text-slate-500 text-2xl" key={item}>
            {item}
          </li>
        ))}
      </ul>
      <Form />
      <p className="text-slate-400 text-sm">Fetched from backend server</p>
    </div>
  );
}

export default App;

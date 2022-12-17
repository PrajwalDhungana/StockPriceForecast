import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [items, setItems] = useState([]);

  axios.defaults.baseURL = 'http://127.0.0.1:5000';

  useEffect(() => {
    async function fetchData() {
      const result = await axios('/items');
      setItems(result.data.items);
    }
    fetchData();
  }, []);

  return (
    <div>
      <h1>Items</h1>
      <ul>
        {items.map(item => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;
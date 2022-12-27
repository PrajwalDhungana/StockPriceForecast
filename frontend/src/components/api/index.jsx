export default class APIService {
  // Insert the tickerSymbol
  static InsertTickerSymobl(body) {
    return fetch(`http://localhost:5000/insertTicker`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ 'tickerSymbol' : ['AAPL']}),
    })
      .then((response) => response.json())
      .catch((error) => console.log(error));
  }
}

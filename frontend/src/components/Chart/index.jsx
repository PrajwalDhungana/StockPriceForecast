import React from "react";
import Chart from "react-apexcharts";

const StockChart = (props) => {
  const date = props.date;
  const close = props.close;
  const data = date.map((date, index) => ({
    x: date,
    y: close[index],
  }));

  const trainDate = props.trainDate;
  const trainClose = props.trainClose;
  const trainData = trainDate.map((date, index) => ({
    x: date,
    y: trainClose[index],
  }));

  const testDate = props.testDate;
  const testClose = props.testClose;
  const testData = testDate.map((date, index) => ({
    x: date,
    y: testClose[index],
  }));

  const predictionDate = props.predictionDate;
  const predictionClose = props.predictionClose;
  const predictionData = predictionDate.map((date, index) => ({
    x: date,
    y: predictionClose[index],
  }));

  const variant = {
    series: [
      {
        name: props.ticker,
        data: data,
      },
      {
        name: "Train",
        data: trainData,
      },
      {
        name: "Test",
        data: testData,
      },
      {
        name: "Prediction",
        data: predictionData,
      },
    ],
    options: {
      chart: {
        type: "line",
        stacked: false,
        zoom: {
          type: "x",
          enabled: true,
          autoScaleYaxis: true,
        },
        toolbar: {
          autoSelected: "zoom",
        },
      },
      colors: ["#FB5607", "#19647E", "#28AFB0", "#036016"],
      stroke: {
        width: 2,
        strokeColor: "#8d5ee7",
      },
      dataLabels: {
        enabled: false,
      },
      markers: {
        size: 0,
      },
      yaxis: {
        labels: {
          formatter: function (val) {
            return "$" + Number(Math.abs(val).toPrecision(5));
          },
          title: {
            formatter: (seriesName) => seriesName,
          },
        },
        title: {
          text: "Price",
        },
      },
      xaxis: {
        type: "datetime",
      },
      legend: {
        position: 'top',
        horizontalAlign: 'center',
        fontSize: '14px',
        floating: false,
        markers: {
          width: 12,
          height: 12,
          radius: '50%',
          onClick: undefined,
          offsetX: -2,
          offsetY: 0
        },    
        itemMargin: {
          horizontal: 10,
          vertical: 0
        }
      },
      tooltip: {
        shared: true,
        marker: {
          show: true,
        },
      },
      responsive: [
        {
          breakpoint: 576,
          options: {
            legend: {
              offsetY: -30,
            },
          },
        },
      ],
    },
  };

  return (
    <div className="chart">
      <Chart
        options={variant.options}
        series={variant.series}
        type="line"
        height={500}
      />
    </div>
  );
};

export default StockChart;

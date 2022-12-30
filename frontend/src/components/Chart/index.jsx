import React from "react";
import Chart from "react-apexcharts";

const StockChart = () => {
  const variant = {
    series: [
      {
        name: "AAPL",
        data: [
          {
            x: new Date("2018-03-12").getTime(),
            y: 76,
          },
          {
            x: new Date("2018-03-13").getTime(),
            y: 78,
          },
          {
            x: new Date("2018-03-14").getTime(),
            y: 79,
          },
          {
            x: new Date("2018-03-15").getTime(),
            y: 69,
          },
          {
            x: new Date("2018-03-16").getTime(),
            y: 83,
          },
          {
            x: new Date("2018-03-17").getTime(),
            y: 88,
          },
          {
            x: new Date("2018-03-18").getTime(),
            y: 81,
          },
          {
            x: new Date("2018-03-19").getTime(),
            y: 79,
          },
          {
            x: new Date("2018-03-20").getTime(),
            y: 83,
          },
        ],
      },
    ],
    options: {
      chart: {
        type: "area",
        stacked: false,
        height: 350,
        width: 700,
        zoom: {
          type: "x",
          enabled: true,
          autoScaleYaxis: true,
        },
        toolbar: {
          autoSelected: "zoom",
        },
      },
      dataLabels: {
        enabled: false,
      },
      markers: {
        size: 0,
      },
      title: {
        text: "Stock Price Movement",
        align: "left",
      },
      fill: {
        type: "gradient",
        gradient: {
          shadeIntensity: 1,
          inverseColors: false,
          opacityFrom: 0.5,
          opacityTo: 0,
          stops: [0, 90, 100],
        },
      },
      yaxis: {
        labels: {
          formatter: function (val) {
            return "$ " + (val).toFixed(2);
          },
          title: {
            formatter: (seriesName) => seriesName + "akldjalskjd",
          },
        },
        title: {
          text: "Price",
        },
      },
      xaxis: {
        type: "datetime",
      },
      tooltip: {
        theme: "dark",
        shared: false,
        y: {
          formatter: function (val) {
            return (val).toFixed(0);
          },
          title: {
            formatter: (seriesName) => seriesName,
        },
        },
      },
    },
  };

  return (
    <div id="chart">
      <Chart
        options={variant.options}
        series={variant.series}
        type="area"
        height={350}
        width={variant.options.chart.width}
      />
    </div>
  );
};

export default StockChart;

import React from "react";
import Chart from "react-apexcharts";

const StockChart = (props) => {
  const key1 = 'x';
  const key2 = 'y';
  const values1 = props.date;
  const values2 = props.close;
  const datas = values1.map((value1, index) => ({ [key1]: value1, [key2]: values2[index] }))

  const variant = {
    series: [
      {
        name: "AAPL",
        data: datas,
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
            return "$ " + val.toFixed(2);
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
            return val.toFixed(0);
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

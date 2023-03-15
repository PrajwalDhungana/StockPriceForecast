import React from "react";
import Chart from "react-apexcharts";

const StockChart = (props) => {
  const key1 = "x";
  const key2 = "y";
  const values1 = props.date;
  const values2 = props.close;
  const datas = values1.map((value1, index) => ({
    [key1]: value1,
    [key2]: values2[index],
  }));

  const variant = {
    series: [
      {
        name: props.ticker,
        data: datas,
      },
      {
        name: "Train",
        data: datas,
      },
    ],
    options: {
      chart: {
        type: "area",
        stacked: false,
        width: 1000,
        zoom: {
          type: "x",
          enabled: true,
          autoScaleYaxis: true,
        },
        toolbar: {
          autoSelected: "zoom",
        },
      },
      colors: ["#1507bb", "#fface2"],
      stroke: {
        width: 1,
        strokeColor: "#8d5ee7",
      },
      dataLabels: {
        enabled: false,
      },
      markers: {
        size: 0,
      },
      title: {
        text: props.ticker + " " + props.message,
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
            return "$" + val.toFixed(2);
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
      tooltip: {
        shared: false,
        y: {
          formatter: function (val) {
            return val.toFixed(2);
          },
          title: {
            formatter: (seriesName) => seriesName,
          },
        },
      },
    },
  };

  return (
    <div className="chart mb-20">
      <Chart
        options={variant.options}
        series={variant.series}
        type="area"
        height={500}
        width={variant.options.chart.width}
      />
    </div>
  );
};

export default StockChart;
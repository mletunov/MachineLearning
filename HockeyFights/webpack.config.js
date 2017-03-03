var webpack = require('webpack');
var path = require('path');
module.exports = {
    entry: [
      "./web/static/scripts/app/app.js"
    ],
    output: {
        path: __dirname + '/web/static',
        filename: "bundle.js"
    },
    resolve: {
        modules: [
          path.join(__dirname, ''),
          'node_modules'
        ],
        extensions: ['.webpack.js', '.web.js', '.js', '.jsx']
    },
    module: {
        loaders: [
          {
              test: /\.js?$/,
              loader: 'babel-loader',
              query: {
                  presets: ['es2015', 'react']
              },
              exclude: /node_modules/
          },
          {
              test: /\.css?$/,
              loader: "style-loader!css-loader"
          }
        ]
    },
    plugins: [
    ]
};
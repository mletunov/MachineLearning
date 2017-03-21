
var path = require('path')
var webpack = require('webpack')
var HtmlWebpackPlugin = require('html-webpack-plugin');
var packageJson = require("../package.json");


module.exports = function (_path, NODE_ENV = "development") {

    return {
        entry: [
            'babel-polyfill',
            './app/app.js'
        ],
        output: {
            path: path.join(_path, 'build'),
            filename: 'bundle.js',
            publicPath: '/'
        },
        module: {
            loaders: [
                {
                    loaders: ['babel-loader'],
                    include: [
                        path.resolve(_path, "app"),
                    ],
                    test: /\.js$/,
                },
                {
                    test: /\.css$/,
                    loader: "style-loader!css-loader"
                },
                {
                    test: /\.scss$/,
                    loader: "style-loader!css-loader!sass-loader"
                },
                {
                    test: /\.(png|jpg|gif)$/,
                    loader: 'url-loader'
                },
                {
                    test: /\.(woff|woff2)(\?v=\d+\.\d+\.\d+)?$/,
                    loader: 'url-loader?limit=10000&mimetype=application/font-woff'
                },
                {
                    test: /\.ttf(\?v=\d+\.\d+\.\d+)?$/,
                    loader: 'url-loader?limit=10000&mimetype=application/octet-stream'
                },
                {
                    test: /\.eot(\?v=\d+\.\d+\.\d+)?$/,
                    loader: 'file-loader'
                },
                {
                    test: /\.svg(\?v=\d+\.\d+\.\d+)?$/,
                    loader: 'url-loader?limit=10000&mimetype=image/svg+xml'
                }
            ]
        },
        plugins: [
            new HtmlWebpackPlugin({
                filename: 'index.html',
                template: _path + '/index.html'
            }),
            new webpack.DefinePlugin({
                __VERSION__: JSON.stringify(packageJson.version),
                "process.env": {
                    NODE_ENV: JSON.stringify(NODE_ENV)
                }
            })
        ]
    }
}

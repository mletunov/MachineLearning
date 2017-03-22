var webpack = require("webpack");

module.exports = function (_path) {

    return {
        devtool: 'cheap-source-map',
        plugins: [
            new webpack.optimize.UglifyJsPlugin({
                compress: {
                    warnings: false
                }
            })
        ]        
    }
};
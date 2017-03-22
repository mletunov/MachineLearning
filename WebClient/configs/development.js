module.exports = function (_path) {

    return {
        devtool: 'source-map',
        devServer: {
            historyApiFallback: true,
            proxy: {
                "/api": {
                    target: "http://localhost:5555",
                    pathRewrite: { "^/api": "" }
                }
            }
        },
    }
};

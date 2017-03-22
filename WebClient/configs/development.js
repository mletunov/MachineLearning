module.exports = function (_path) {

    return {
        devtool: 'source-map',
        devServer: {
            historyApiFallback: true,
            proxy: [{
                path: "/api",
                target: "http://localhost:5555",
                ignorePath: true
            }]
        },
    }
};

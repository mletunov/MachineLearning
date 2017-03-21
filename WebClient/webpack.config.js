
var merge = require("webpack-merge");

var _configs = {
    global: require(__dirname + "/configs/global"),
    development: require(__dirname + "/configs/development"),
    production: require(__dirname + "/configs/production")
}

var _loadConfiguration = function (environment) {

    if (!environment) throw Exception("Can't find environment variable");
    if (!_configs[environment]) throw Exception("Can't find environment configuration. See _configs object");

    return merge(
        _configs["global"](__dirname, environment),
        _configs[environment](__dirname, environment)
    );

}

module.exports = _loadConfiguration(process.env.NODE_ENV);

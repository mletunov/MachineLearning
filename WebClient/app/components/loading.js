import React, { Component } from 'react';

class Loading extends Component {

    constructor(props) {
        super(props);
    }



    render() {
        return (
            <div className={"loading " + (this.props.isLoading ? "active-loading" : "")}>
                {this.props.isLoading ?
                    <div className={this.props.isBig ? "big-spinner" : "small-spinner"} /> :
                    this.props.children
                }
            </div>
        );
    }
}



export default Loading;

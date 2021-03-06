﻿import React, { Component } from 'react';
import Header from './header';
import Footer from './footer';
import {logSrv} from '../helpers';

class Main extends Component {

    constructor(props) {
        super(props);
    }

    componentWillMount(){
      window.onerror = () => logSrv.Error();
    }

    render() {
        return (
            <div>
                <Header/>
                  <div className="container content-block">
                      {this.props.children}
                  </div>
                <Footer/>
            </div>
        );
    }
}


export default Main;

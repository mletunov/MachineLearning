import React from 'react';
import { render } from 'react-dom';
import { Router, Route, browserHistory, IndexRoute} from 'react-router';
import { Provider } from 'react-redux';
import reduxStore from './store/reduxStore';
import reducers from './reducers/index';
import main from './components/main';
import uploadPage from './components/uploadPage';
import videoPage from './components/videoPage';
import 'bootstrap/dist/css/bootstrap.css';
import "./assets/styles/site.css";
import "video-react/dist/video-react.css";

render(
  <Provider store={reduxStore}>
    <Router history={browserHistory}>
        <Route path="/" component={main}>
          <IndexRoute component={uploadPage} />
          <Route path="deeplink/:sessionId" component={videoPage}/>
        </Route>
    </Router>
  </Provider>,
  document.getElementById('app'));

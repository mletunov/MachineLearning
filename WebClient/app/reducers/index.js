import { combineReducers } from 'redux';
import commonReducer from './commonReducer';
import loadingReducer from './loadingReducer';


const reducers = combineReducers({
    commonReducer,
    loadingReducer
})

export default reducers

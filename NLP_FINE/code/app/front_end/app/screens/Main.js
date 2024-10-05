import React, { useContext } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Context } from '../utils/Context';
import TabNavigator from '../components/TabNavigator';
import Loading from './Loading';

const Stack = createNativeStackNavigator();

const Main = () => {
	const { isLogin } = useContext(Context);

	return !isLogin ? (
		<Loading />
	) : (
		<NavigationContainer>
			<Stack.Navigator initialRouteName='ChatBot'>
				<Stack.Screen name='Root' component={TabNavigator} options={{headerShown: false}} />
			</Stack.Navigator>
		</NavigationContainer>
	);
};

export default Main;
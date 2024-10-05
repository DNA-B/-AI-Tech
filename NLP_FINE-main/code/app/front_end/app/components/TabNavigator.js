import { StatusBar } from 'react-native';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import styled, { useTheme } from 'styled-components/native';
import { w64, w92, w160 } from '../utils/theme';
import ChatBot from '../screens/ChatBot';
import Feed from '../screens/Feed';

const Tab = createMaterialTopTabNavigator();

const TabNavigator = () => {
	const theme = useTheme();

	return (
		<Container>
			<StatusBar translucent backgroundColor='transparent' />
			<Tab.Navigator
				theme={theme}
				initialRouteName='ChatBot'
				screenOptions={{
					tabBarShowIcon: true,
					tabBarShowLabel: false,
					swipeEnabled: false,
					tabBarStyle: {
						height: w160,
						backgroundColor: 'transparent',
						position: 'absolute',
						top: w92,
						left: 0,
						right: 0,
						paddingTop: 4
					},
					tabBarIndicatorStyle: {
						backgroundColor: theme.background
					}
			}}>
				<Tab.Screen
					name='ChatBot'
					component={ChatBot}
					options={{
						headerShown: false,
						tabBarIcon: ({ focused }) => (
							<Icon source={focused ? require('../assets/chat-active-icon.png')
								: require('../assets/chat-icon.png')} />
						)
					}}
				/>
				<Tab.Screen
					name='Feed'
					component={Feed}
					options={{
						headerShown: false,
						tabBarIcon: ({ focused }) => (
							<Icon source={focused ? require('../assets/feed-active-icon.png')
								: require('../assets/feed-icon.png')} />
						)
					}}
				/>
			</Tab.Navigator>
		</Container>
	)
};

const Container = styled.View`
	flex: 1;
`;

const Icon = styled.Image`
	width: ${w64}px;
	height: ${w64}px;
`;

export default TabNavigator;
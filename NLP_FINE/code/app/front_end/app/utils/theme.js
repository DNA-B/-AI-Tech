import { Dimensions } from 'react-native';

export const getTheme = () => {
	const morningTheme = {
		primary: '#F6CB42',
		secondary: '#FF8562',
		primaryBackground: '#F6D365',
		secondaryBackground: '#FDA085',
		background: '#FFFDF9'
	}
  
	const afternoonTheme = {
		primary: '#ACE56D',
		secondary: '#3FA3F4',
		primaryBackground: '#C2E59C',
		secondaryBackground: '#64B3F4',
		background: '#F9FFFD'
	}
  
	const eveningTheme = {
		primary: '#4568DC',
		secondary: '#B06AB3',
		primaryBackground: '#6886EB',
		secondaryBackground: '#C88ACB',
		background: '#F6F8FF'
	}
  
	const nightTheme = {
		primary: '#2B5876',
		secondary: '#4E4376',
		primaryBackground: '#497898',
		secondaryBackground: '#695E92',
		background: '#F6FCFF'
	}

	const now = new Date();
	const hour = now.getHours();
	if (hour <= 6) {
		theme = nightTheme;
	}
	else if (hour <= 12) {
		theme = morningTheme;
	}
	else if (hour <= 18) {
		theme = afternoonTheme;
	}
	else {
		theme = eveningTheme;
	}

	return {
		...theme,
		gray: '#DADCE0'
	};
}

export const SCREEN_WIDTH = Dimensions.get('window').width;

// device의 width를 1080px을 기준으로 할 때, 해당하는 px의 값
export const w8 = SCREEN_WIDTH * 8 / 1080;
export const w14 = SCREEN_WIDTH * 14 / 1080;
export const w16 = SCREEN_WIDTH * 16 / 1080;
export const w24 = SCREEN_WIDTH * 24 / 1080;
export const w28 = SCREEN_WIDTH * 28 / 1080;
export const w32 = SCREEN_WIDTH * 32 / 1080;
export const w48 = SCREEN_WIDTH * 48 / 1080;
export const w64 = SCREEN_WIDTH * 64 / 1080;
export const w84 = SCREEN_WIDTH * 84 / 1080;
export const w92 = SCREEN_WIDTH * 92 / 1080;
export const w96 = SCREEN_WIDTH * 96 / 1080;
export const w108 = SCREEN_WIDTH * 108 / 1080;
export const w144 = SCREEN_WIDTH * 144 / 1080;
export const w160 = SCREEN_WIDTH * 160 / 1080;

export const androidHeader = SCREEN_WIDTH * 252 / 1080;
export const iosHeader = SCREEN_WIDTH * 160 / 1080;
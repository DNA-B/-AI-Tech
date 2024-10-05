import { Platform } from 'react-native';
import styled from 'styled-components/native';
import { LinearGradient } from 'expo-linear-gradient';
import { w96, androidHeader, iosHeader } from '../utils/theme';

export default Header = styled(LinearGradient).attrs(({ view, theme }) => ({
	colors: view === 'chatbot' ? 
		[theme.secondaryBackground, theme.primaryBackground]
		: [theme.primaryBackground, theme.secondaryBackground],
	start: { x: 0, y: 0 },
	end: { x: 1, y: 1 }
}))`
	height: ${Platform.OS === 'android' ? androidHeader : iosHeader}px;
	padding-top: ${Platform.OS === 'android' ? w96 : 0}px;
`;
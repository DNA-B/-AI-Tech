import Toast from 'react-native-root-toast';

export const toast = (str) => {
    Toast.show(str, {
        duration: Toast.durations.SHORT,
        position: -100,
        opacity: 0.8,
        backgroundColor: '#222222',
        containerStyle: {
            borderRadius: 100,
            paddingHorizontal: 16
        }
    });
};